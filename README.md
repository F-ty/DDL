# ITFD

ITFD 是一个用于组合图像检索（Composed Image Retrieval, CIR）的模型实现。CIR 的查询由参考图像和修改文本组成，目标是在候选图像库中检索符合修改要求的目标图像。

一个样本可表示为：

```text
<reference image Ir, modification text Tm, target image It>
```

例如，参考图像是一件黑色短裙，修改文本为“改成更长的白色裙子”，模型需要保留“裙子”等共同属性，弱化或去除“黑色、短”等参考属性，并加入“白色、长”等目标属性。ITFD 的目标是在 CLIP 公共特征空间中生成组合查询特征 `Fq`，使其接近目标图像特征 `Ft`。

## 方法概述

常规 CIR 方法通常直接融合完整参考图像特征和修改文本特征：

```text
Fq = Fusion(Fr, Fm)
```

这种方式可能把参考图像中应被删除的属性继续保留到查询特征中。例如文本要求“白色衬衫改成红色”时，直接融合可能仍保留“白色”特征。

ITFD 的核心思想是：

```text
先解耦，再融合，最后进行目标图像匹配。
```

整体上，模型希望从修改文本中区分删除、保留、添加语义，并从参考图像中提取需要保留到目标图像中的视觉信息。最终用于检索的主要是：

```text
参考图像保留特征 + 文本添加特征
```

## 数据流

本实现的主要数据流如下：

```text
参考图像 Ir
  -> CLIP image encoder
  -> 参考图像特征 Fr

目标图像 It
  -> CLIP image encoder
  -> 目标图像特征 Ft

参考图像描述 + 修改文本
  -> CLIP text encoder
  -> 增强修改文本特征 Fm

Fm
  -> 文本语义解耦
  -> 删除掩码 / 保留掩码 / 添加掩码

保留掩码 + Fr
  -> 文本条件化图像保留特征 Fr-prs

添加掩码 + Fm
  -> 文本添加特征 Fm-add

Fm-add + Fr-prs
  -> 自适应动态融合
  -> 组合查询特征 Fq

Fq 与 Ft
  -> 批次级对比学习和检索评测
```

核心伪代码：

```python
ref = normalize(CLIP_image(reference_image))
target = normalize(CLIP_image(target_image))
text = normalize(CLIP_text(reference_caption + ", but " + modification_text))

delete_mask = sigmoid(FC_delete(text))
preserve_mask = sigmoid(FC_preserve(text))
add_mask = sigmoid(FC_add(text))

delete_text = normalize(delete_mask * text)
add_text = normalize(add_mask * text)
preserved_ref = normalize(preserve_mask * ref)
preserved_target = normalize(preserve_mask * target)

weight = sigmoid(FusionFC(concat(add_text, preserved_ref)))

query = normalize(
    weight * add_text
    + (1 - weight) * preserved_ref
)

ranking_loss = CrossEntropy(scale * query @ target.T, diagonal_labels)
triplet_loss = Triplet(delete_text, ref, target) + Triplet(add_text, target, ref)
preserve_loss = mean(1 - cosine(preserved_ref, preserved_target))

total_loss = ranking_loss + alpha * triplet_loss + beta * preserve_loss
```

## 编码器

本项目使用 OpenCLIP 作为统一图文编码器：

```python
open_clip.create_model_and_transforms(args.backbone, pretrained=args.pt_path)
```

图像编码：

```python
self.clip.encode_image(x)
```

文本编码：

```python
self.tokenizer(txt)
self.clip.encode_text(txt)
```

图像、文本和查询特征均使用 L2 归一化，因此检索时可以用点积计算余弦相似度。

训练脚本支持的骨干网络配置包括：

```text
ViT-B-32: hidden_dim = 512
ViT-H-14: hidden_dim = 1024
RN50 或其他分支: hidden_dim 通常按 512 配置
```

训练默认使用两组学习率：

```text
CLIP 编码器: 1e-6
ITFD 新增模块: 1e-4
```

## 文本构造

原始修改文本通常只描述变化，缺少参考图像完整语义。例如：

```text
is longer and white instead of black
```

因此数据集代码会把参考图像描述和修改文本拼接为增强文本：

```text
a woman in black one shoulder dress, but is longer and white instead of black
```

在本实现中，`textual_query` 通常不是单独的原始修改文本，而是：

```text
reference caption + ", but " + modification text
```

FashionIQ 和 Shoes 使用预生成参考图像描述补充修改文本；Fashion200K 会根据参考图像描述和目标图像描述之间的词差异构造类似 `replace black with blue` 的修改文本，并在评测中拼接参考描述。

## 模型模块

### 文本语义解耦

`src/model.py` 中使用三个独立线性映射和 Sigmoid 生成逐维掩码：

```python
self.del_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
self.prs_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
self.new_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
```

对应语义为：

```text
del_mask: 删除语义，例如 black、short、long sleeves
prs_mask: 保留语义，例如 dress、shirt、shoe
new_mask: 添加语义，例如 white、longer、with buckle
```

主查询分支使用：

```text
new_text = normalize(new_mask * textual_query)
prs_ref = normalize(prs_mask * visual_query)
```

`del_mask` 主要通过三元组损失参与训练，`prs_mask` 用于提取参考图像和目标图像中的保留部分，`new_mask` 用于得到添加文本特征。

### 文本引导的图像保留特征提取

文本生成的保留掩码作用于 CLIP 图像全局特征，用于提取参考图像中需要保留到目标图像中的视觉信息：

```python
prs_ref = F.normalize(prs_mask * visual_query, p=2, dim=-1)
```

### 自适应动态图文融合

参与融合的特征包括：

```text
new_text: 添加文本特征
prs_ref: 参考图像保留特征
```

模型先拼接两者，再预测动态权重：

```python
combined_feature = self.combiner_fc(torch.cat([new_text, prs_ref], dim=-1))
dynamic_scaler = self.scaler_fc(self.dropout(combined_feature))
query = dynamic_scaler * new_text + (1 - dynamic_scaler) * prs_ref
query = F.normalize(query, p=2, dim=-1)
```

在本实现中，`dynamic_scaler` 乘在 `new_text` 上，因此对应文本添加特征的权重。

## 损失函数

总体损失由三部分组成：

```text
L = Lranking + alpha * Ltrip + beta * Lcos
```

### 检索排序损失

查询特征与 batch 内目标图像特征计算相似度矩阵：

```python
x = torch.mm(query, target.t())
```

对于 batch size 为 `B` 的批次，标签为：

```python
labels = [0, 1, 2, ..., B-1]
```

模型使用交叉熵进行批次级对比学习：

```python
loss = F.cross_entropy(self.loss_weight * x, labels)
```

其中 `loss_weight` 是可学习相似度缩放参数，初始值为 10。

### 文本解耦三元组损失

删除文本特征应更接近参考图像并远离目标图像：

```python
con1 = self.trip(del_text, ref, target)
```

添加文本特征应更接近目标图像并远离参考图像：

```python
con2 = self.trip(new_text, target, ref)
```

三元组损失 margin 为 1.0。

### 保留特征余弦损失

相同保留掩码会分别施加到参考图像和目标图像特征上：

```python
prs_ref = normalize(prs_mask * ref)
prs_tar = normalize(prs_mask * target)
loss3 = mean(1.0 - cosine_similarity(prs_ref, prs_tar))
```

该损失鼓励参考图像和目标图像的保留属性在特征空间中一致。

### 损失权重

`ITFD_GitHub/src/model.py` 中的默认损失为：

```python
return loss + 0.2 * loss2 + 0.7 * loss3
```

对应默认配置为：

```text
alpha = 0.2
beta = 0.7
```

`alpha` 和 `beta` 是辅助损失的超参数，可按数据集调整。实验中可在 `0.1` 到 `1.0` 之间按一位小数搜索，例如 `0.1, 0.2, ..., 1.0`。不同数据集的最优权重可能不同，复现实验时应记录 `src/model.py`、训练命令和日志中的实际参数。

## 复现说明

复现实验时应保持以下配置一致：

1. `dynamic_scaler` 对应文本添加特征权重。
2. 损失权重 `alpha` 和 `beta` 默认为 `alpha=0.2, beta=0.7`，不同数据集可使用不同配置。
3. 对比实验需记录数据集划分、backbone、损失权重、checkpoint 和训练日志。

## 代码结构

```text
ITFD_GitHub/
├── src/
│   ├── model.py       # ITFD 模型、CLIP 编码、掩码解耦、动态融合、损失
│   ├── datasets.py    # FashionIQ、Shoes、Fashion200K 数据读取和文本构造
│   ├── train.py       # 参数解析、训练、验证、checkpoint 保存
│   ├── eval.py        # checkpoint 加载和评测入口
│   ├── test.py        # Recall 计算和检索评测逻辑
│   └── utils.py       # 日志、JSON、checkpoint 工具
├── tools/             # ITFD 效率 profiling
├── data/              # 数据集和 OpenCLIP 预训练权重
├── scripts/           # 根目录可运行脚本
└── requirements.txt
```

## 环境

推荐环境为 Python 3.8 和 PyTorch。一个典型 CUDA 环境如下：

```bash
conda create -n itfd python=3.8 -y
conda activate itfd
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

CUDA 版本不同时，请先安装匹配的 PyTorch 版本，再安装 `requirements.txt`。

## 数据

数据目录布局如下：

```text
data/
├── FashionIQ/
│   ├── captions/
│   ├── image_splits/
│   └── resized_images/
├── Shoes/
│   ├── attributedata/
│   └── captions/
├── Fashion200k/
│   ├── detection/
│   ├── labels/
│   ├── women/
│   └── test_queries.txt
└── pretrain/
    └── CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin
```

数据和预训练权重包含大文件。需要发布这些文件时，可使用 Git LFS：

```bash
git lfs install
git lfs track "*.bin" "*.pt" "*.pth" "*.jpg" "*.jpeg" "*.png"
```

`.gitattributes` 已包含这些规则。

## 训练

从仓库根目录运行：

```bash
bash scripts/train.sh dress
bash scripts/train.sh shirt
bash scripts/train.sh toptee
bash scripts/train.sh shoes
bash scripts/train.sh fashion200k
```

脚本默认配置：

```text
backbone: ViT-B-32
batch size: 64
num_workers: 6
output: outputs/<dataset>/ViT-B-32/
```

等价直接命令：

```bash
python src/train.py \
  --dataset dress \
  --fashioniq_split val-split \
  --backbone ViT-B-32 \
  --hidden_dim 512 \
  --batch_size 64 \
  --num_workers 6 \
  --model_dir outputs/dress/ViT-B-32
```

训练时每个 batch 的字段含义：

```text
visual_query: 参考图像
textual_query: 参考图像描述 + ", but " + 修改文本
target_img_data: 目标图像
```

`train.py` 会将 CLIP 参数和新增模块参数分成不同学习率的参数组，并使用 AdamW 和混合精度训练。

## 评测

将 ITFD checkpoint 放入 `checkpoints/` 后运行：

```bash
bash scripts/eval.sh dress checkpoints/dress_0_best_model.pt
```

直接命令：

```bash
python src/eval.py \
  --dataset dress \
  --fashioniq_split val-split \
  --backbone ViT-B-32 \
  --hidden_dim 512 \
  --batch_size 64 \
  --num_workers 6 \
  --ckpt checkpoints/dress_0_best_model.pt
```

FashionIQ 和 Shoes 主要输出：

```text
Recall@1
Recall@10
Recall@50
```

Fashion200K 评测会先编码所有候选图像，再计算查询与候选图像的相似度并排序。复现实验时应保存完整命令、checkpoint、日志和实际 `model.py` 损失权重。

## 效率 Profiling

```bash
python tools/profile_itfd_efficiency_full.py \
  --dataset dress \
  --checkpoint checkpoints/dress_0_best_model.pt \
  --output_json outputs/efficiency/itfd_dress.json \
  --output_csv outputs/efficiency/itfd_dress.csv
```

## 实验结果

论文报告的主要结果包括：

```text
FashionIQ average Recall: 66.21
Shoes average Recall: 74.70
Fashion200K average Recall: 67.09
```

消融实验显示，同时使用文本语义解耦、图像特征解耦、动态融合和辅助解耦损失时效果最好。复现实验应以实际运行配置和日志为准。

## 开发与扩展

扩展模型或添加新实验时，主要涉及以下模块：

1. `src/datasets.py`：数据读取、文本构造、字段定义和张量组织。
2. `src/model.py`：查询特征提取、文本语义解耦、动态图文融合和损失函数。
3. `src/train.py`：训练参数、优化器、学习率设置和 checkpoint 保存。
4. `src/eval.py` 与 `src/test.py`：checkpoint 加载、候选图像编码和检索指标计算。

更换 backbone 时需同步设置 `hidden_dim`。调整损失权重或融合结构时，应记录对应的数据集、训练命令、checkpoint 和日志，确保实验结果可追踪。

## 仓库说明

- `src/exp/` 未包含在发布包中，其中通常包含大量实验输出和 checkpoint。
- `checkpoints/` 和 `outputs/` 不作为空目录保留；训练和 profiling 会按需生成输出目录，评测时可将 checkpoint 放入 `checkpoints/` 或传入自定义路径。
- 辅助可视化、调试和样例挑选脚本不作为核心训练评测入口。
- 数据集、预训练权重、训练日志、输出结果和模型 checkpoint 通常不直接提交到 Git 仓库；如需发布大文件，建议使用 Git LFS。
