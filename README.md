# DDL: A Dual Disentanglement Learning Framework with Intent-Aware Multimodal Fusion for Interactive Image Retrieval

This repository provides the implementation of **DDL**, a **Dual Disentanglement Learning** framework with intent-aware multimodal fusion for **Interactive Image Retrieval (IIR)**.

Interactive Image Retrieval uses a reference image and feedback text to retrieve a target image from a candidate gallery. This setting is widely used in applications such as security monitoring and intelligent e-commerce, where users iteratively refine visual search results through natural-language feedback.

In this task, a query sample can be represented as:

```text
<reference image Ir, feedback text T, target image It>
```

For example, if the reference image is a black short dress and the feedback text says "make it longer and white", the model should retain common semantics such as "dress", suppress or remove outdated reference attributes such as "black" and "short", and add target-oriented semantics such as "white" and "longer". DDL aims to generate a composed query feature `Fq` in the CLIP feature space so that it is close to the target image feature `Ft`.

## Abstract

Interactive Image Retrieval (IIR) integrates reference images and feedback text to retrieve target images. Existing methods often fuse image and text features directly, introducing irrelevant information and limiting retrieval accuracy. To overcome this limitation, we propose **DDL**, a Dual Disentanglement Learning framework with intent-aware multimodal fusion for IIR. Specifically, a **Text Semantic Disentanglement Module (TSDM)** decomposes text into deletion, retention, and addition features, enabling precise semantic extraction. Then, a **Text-Guided Image Disentanglement Module (TGIDM)** selectively isolates target-aligned visual features from the reference image under text guidance. This dual disentanglement ensures that only intention-aware features undergo adaptive fusion. Finally, a multi-objective joint loss function is constructed to achieve more semantically aligned retrieval.

## Method Overview

Conventional IIR/CIR methods usually fuse the complete reference image feature and the complete feedback text feature directly:

```text
Fq = Fusion(Fr, Ftext)
```

This strategy may preserve irrelevant or conflicting information from the reference image. For example, when the text asks to change a white shirt to red, direct fusion may still keep the "white" visual semantics in the final query.

DDL follows a dual disentanglement strategy:

```text
disentangle text semantics, disentangle reference image features under text guidance,
then perform intent-aware multimodal fusion for target image matching.
```

The framework first separates feedback text into deletion, retention, and addition semantics through TSDM. It then uses the retention semantics to guide TGIDM, isolating target-aligned visual features from the reference image. The final retrieval query mainly combines:

```text
reference-image retained visual features + feedback-text addition features
```

## Data Flow

The main data flow of this implementation is:

```text
Reference image Ir
  -> CLIP image encoder
  -> Reference image feature Fr

Target image It
  -> CLIP image encoder
  -> Target image feature Ft

Reference caption + feedback text
  -> CLIP text encoder
  -> Enhanced feedback text feature Fm

Fm
  -> Text Semantic Disentanglement Module (TSDM)
  -> deletion mask / retention mask / addition mask

Retention mask + Fr
  -> Text-Guided Image Disentanglement Module (TGIDM)
  -> Text-guided retained reference feature Fr-prs

Addition mask + Fm
  -> Text addition feature Fm-add

Fm-add + Fr-prs
  -> intent-aware adaptive multimodal fusion
  -> Composed query feature Fq

Fq and Ft
  -> batch-level contrastive learning and retrieval evaluation
```

Core pseudo-code:

```python
ref = normalize(CLIP_image(reference_image))
target = normalize(CLIP_image(target_image))
text = normalize(CLIP_text(reference_caption + ", but " + feedback_text))

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

## Encoder

This project uses OpenCLIP as the unified image-text encoder:

```python
open_clip.create_model_and_transforms(args.backbone, pretrained=args.pt_path)
```

Image encoding:

```python
self.clip.encode_image(x)
```

Text encoding:

```python
self.tokenizer(txt)
self.clip.encode_text(txt)
```

Image, text, and query features are all L2-normalized, so retrieval can use dot product as cosine similarity.

Supported backbone configurations in the training scripts include:

```text
ViT-B-32: hidden_dim = 512
ViT-H-14: hidden_dim = 1024
RN50 or other branches: hidden_dim is usually set to 512
```

The default training setup uses two learning rates:

```text
CLIP encoder: 1e-6
DDL newly added modules: 1e-4
```

## Feedback Text Construction

The original feedback text usually describes only the desired change and lacks the full semantics of the reference image. For example:

```text
is longer and white instead of black
```

Therefore, the dataset code concatenates the reference caption and feedback text into an enhanced text query:

```text
a woman in black one shoulder dress, but is longer and white instead of black
```

In this implementation, `textual_query` is usually not the raw feedback text alone, but:

```text
reference caption + ", but " + feedback text
```

FashionIQ and Shoes use pre-generated reference captions to enrich feedback text. Fashion200K constructs feedback text such as `replace black with blue` from word differences between reference and target captions, and concatenates the reference caption during evaluation.

## Model Modules

### Text Semantic Disentanglement Module (TSDM)

In `src/model.py`, three independent linear projections followed by Sigmoid generate element-wise masks:

```python
self.del_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
self.prs_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
self.new_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
```

Their semantics are:

```text
del_mask: deletion semantics, such as black, short, long sleeves
prs_mask: retention semantics, such as dress, shirt, shoe
new_mask: addition semantics, such as white, longer, with buckle
```

The main query branch uses:

```text
new_text = normalize(new_mask * textual_query)
prs_ref = normalize(prs_mask * visual_query)
```

`del_mask` is mainly optimized through the triplet loss. `prs_mask` is used to extract retained features from both reference and target images. `new_mask` is used to obtain the addition text feature.

### Text-Guided Image Disentanglement Module (TGIDM)

The retention mask generated by TSDM is applied to the global CLIP image feature. This extracts the visual information in the reference image that should be retained in the target image under feedback-text guidance:

```python
prs_ref = F.normalize(prs_mask * visual_query, p=2, dim=-1)
```

### Intent-Aware Adaptive Multimodal Fusion

The features used for fusion are:

```text
new_text: addition text feature
prs_ref: retained reference image feature
```

The model first concatenates the two features and then predicts a dynamic fusion weight:

```python
combined_feature = self.combiner_fc(torch.cat([new_text, prs_ref], dim=-1))
dynamic_scaler = self.scaler_fc(self.dropout(combined_feature))
query = dynamic_scaler * new_text + (1 - dynamic_scaler) * prs_ref
query = F.normalize(query, p=2, dim=-1)
```

In this implementation, `dynamic_scaler` is multiplied with `new_text`, so it corresponds to the weight of the addition text feature.

## Loss Function

The overall loss consists of three terms:

```text
L = Lranking + alpha * Ltrip + beta * Lcos
```

### Retrieval Ranking Loss

The query feature and in-batch target image features are used to compute a similarity matrix:

```python
x = torch.mm(query, target.t())
```

For a batch size of `B`, the labels are:

```python
labels = [0, 1, 2, ..., B-1]
```

The model uses cross-entropy for batch-level contrastive learning:

```python
loss = F.cross_entropy(self.loss_weight * x, labels)
```

Here, `loss_weight` is a learnable similarity scaling parameter initialized to 10.

### Text Disentanglement Triplet Loss

The deletion text feature should be closer to the reference image and farther from the target image:

```python
con1 = self.trip(del_text, ref, target)
```

The addition text feature should be closer to the target image and farther from the reference image:

```python
con2 = self.trip(new_text, target, ref)
```

The triplet loss margin is 1.0.

### Retention Feature Cosine Loss

The same retention mask is applied to both reference and target image features:

```python
prs_ref = normalize(prs_mask * ref)
prs_tar = normalize(prs_mask * target)
loss3 = mean(1.0 - cosine_similarity(prs_ref, prs_tar))
```

This loss encourages retained attributes in the reference and target images to be aligned in the feature space.

### Loss Weights

The default loss in `src/model.py` is:

```python
return loss + 0.2 * loss2 + 0.7 * loss3
```

The corresponding default configuration is:

```text
alpha = 0.2
beta = 0.7
```

`alpha` and `beta` are hyperparameters for the auxiliary losses and can be adjusted by dataset. In experiments, they can be searched from `0.1` to `1.0` with one decimal place, such as `0.1, 0.2, ..., 1.0`. The optimal weights may differ across datasets. For reproducibility, record the actual `src/model.py`, training command, and logs used for each run.

## Reproducibility Notes

To reproduce experiments, keep the following configuration details consistent:

1. `dynamic_scaler` corresponds to the weight of the addition text feature.
2. The default loss weights are `alpha=0.2` and `beta=0.7`; different datasets may use different configurations.
3. Comparative experiments should record the dataset split, backbone, loss weights, checkpoint, and training logs.

## Repository Structure

```text
ITFD_GitHub/
├── src/
│   ├── model.py       # DDL implementation, CLIP encoding, dual disentanglement, intent-aware fusion, loss
│   ├── datasets.py    # FashionIQ, Shoes, and Fashion200K data loading and text construction
│   ├── train.py       # Argument parsing, training, validation, and checkpoint saving
│   ├── eval.py        # Checkpoint loading and evaluation entry point
│   ├── test.py        # Recall computation and retrieval evaluation logic
│   └── utils.py       # Logging, JSON, and checkpoint utilities
├── tools/             # DDL efficiency profiling
├── data/              # Datasets and OpenCLIP pretrained weights
├── scripts/           # Runnable scripts from the repository root
└── requirements.txt
```

## Environment

Python 3.8 and PyTorch are recommended. A typical CUDA environment is:

```bash
conda create -n ddl python=3.8 -y
conda activate ddl
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If your CUDA version differs, install the matching PyTorch version first, then install `requirements.txt`.

## Data

The expected data directory layout is:

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

Datasets and pretrained weights contain large files. If these files need to be released, Git LFS can be used:

```bash
git lfs install
git lfs track "*.bin" "*.pt" "*.pth" "*.jpg" "*.jpeg" "*.png"
```

The `.gitattributes` file already contains these rules.

## Training

Run from the repository root:

```bash
bash scripts/train.sh dress
bash scripts/train.sh shirt
bash scripts/train.sh toptee
bash scripts/train.sh shoes
bash scripts/train.sh fashion200k
```

Default script configuration:

```text
backbone: ViT-B-32
batch size: 64
num_workers: 6
output: outputs/<dataset>/ViT-B-32/
```

Equivalent direct command:

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

Fields in each training batch:

```text
visual_query: reference image
textual_query: reference caption + ", but " + feedback text
target_img_data: target image
```

`train.py` separates CLIP parameters and newly added module parameters into different learning-rate groups, and trains with AdamW and mixed precision.

## Evaluation

Place the DDL checkpoint in `checkpoints/` and run:

```bash
bash scripts/eval.sh dress checkpoints/dress_0_best_model.pt
```

Direct command:

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

FashionIQ and Shoes mainly report:

```text
Recall@1
Recall@10
Recall@50
```

Fashion200K evaluation first encodes all candidate images, then ranks candidates by similarity to each query. For reproducibility, save the full command, checkpoint, logs, and the actual loss weights used in `model.py`.

## Efficiency Profiling

```bash
python tools/profile_itfd_efficiency_full.py \
  --dataset dress \
  --checkpoint checkpoints/dress_0_best_model.pt \
  --output_json outputs/efficiency/ddl_dress.json \
  --output_csv outputs/efficiency/ddl_dress.csv
```

The profiling script name currently retains an earlier implementation name. Use the file name that exists in this repository as the runnable entry point.

## Experimental Results

The experiments show that DDL achieves new state-of-the-art performance on multiple IIR benchmark datasets, especially on the `R@10` and `R@50` metrics, and demonstrates strong robustness.

```text
Primary metrics: Recall@10, Recall@50
Main conclusion: DDL achieves new state-of-the-art retrieval performance and robust results.
```

Ablation studies show that the best performance is obtained when TSDM, TGIDM, intent-aware adaptive fusion, and the multi-objective joint loss are used together. Reproducibility should be based on the actual run configuration, checkpoint, and logs.

## Development and Extension

When extending the model or adding new experiments, the main modules are:

1. `src/datasets.py`: data loading, text construction, field definitions, and tensor organization.
2. `src/model.py`: query feature extraction, text semantic disentanglement, text-guided image disentanglement, intent-aware fusion, and loss functions.
3. `src/train.py`: training arguments, optimizer, learning-rate settings, and checkpoint saving.
4. `src/eval.py` and `src/test.py`: checkpoint loading, candidate image encoding, and retrieval metric computation.

When changing the backbone, update `hidden_dim` accordingly. When adjusting loss weights or the fusion structure, record the dataset, training command, checkpoint, and logs to keep experiments traceable.

## Repository Notes

- `src/exp/` is not included in the release package because it usually contains many experiment outputs and checkpoints.
- The class name in `src/model.py` is currently kept as `ITFD` for compatibility with existing training and evaluation code; the README and paper method name are standardized as DDL.
- `checkpoints/` and `outputs/` are not preserved as empty directories. Training and profiling create output directories as needed. For evaluation, place checkpoints in `checkpoints/` or pass a custom checkpoint path.
- Auxiliary visualization, debugging, and sample selection scripts are not part of the core training and evaluation entry points.
- Datasets, pretrained weights, training logs, output results, and model checkpoints are usually not committed directly to the Git repository. Use Git LFS if large files need to be released.
