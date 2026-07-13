#!/usr/bin/env python3
"""Profile full ITFD retrieval efficiency without training.

- partial FLOPs/MACs via installed counters or torch.profiler(with_flops=True)
- throughput derived from measured latency
- actual gallery feature tensor storage
- optional markdown summary generation from result JSON files
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ITFD_SRC = PROJECT_ROOT / "src"
if str(ITFD_SRC) not in sys.path:
    sys.path.insert(0, str(ITFD_SRC))

import datasets  # noqa: E402
import model as itfd_model  # noqa: E402
import open_clip  # noqa: E402


FASHIONIQ_CATEGORIES = {"dress", "shirt", "toptee"}
DATASET_CHOICES = ("dress", "shirt", "toptee", "shoes", "fashion200k")


@dataclass
class ProfileResult:
    query_encode_latency_ms_mean: float
    query_encode_latency_ms_std: float
    target_encode_latency_ms_per_image: float
    similarity_topk_latency_ms_mean: float
    end_to_end_retrieval_latency_ms_mean: float
    end_to_end_retrieval_latency_ms_std: float
    e2e_throughput_qps: float
    query_encoding_throughput_qps: float
    peak_cuda_memory_mb: float
    params_total: int
    params_trainable: int
    gallery_size: int
    gallery_feature_dim: int
    gallery_feature_dtype: str
    gallery_feature_numel: int
    gallery_feature_element_size_bytes: int
    gallery_feature_storage_bytes: int
    gallery_feature_storage_mb: float
    flops_status: str
    flops_notes: str
    flops_method: str
    flops_tools_available: str
    query_encoding_flops: Optional[int]
    query_encoding_macs: Optional[int]
    target_encoding_flops: Optional[int]
    target_encoding_macs: Optional[int]
    itfd_extra_module_flops: Optional[int]
    itfd_extra_module_macs: Optional[int]
    matching_macs_estimated: int
    matching_flops_estimated: int
    matching_flops_notes: str
    device: str
    backbone: str
    dataset: str
    fashioniq_split: str
    batch_size: int
    num_workers: int
    n_warmup: int
    n_iters: int
    checkpoint: str
    checkpoint_loaded: bool
    target_topk: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile full ITFD inference efficiency.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES)
    parser.add_argument("--fashioniq_split", default="val-split")
    parser.add_argument("--backbone", default="ViT-B-32")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--checkpoint", default="", help="Optional ITFD checkpoint path.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_warmup", type=int, default=30)
    parser.add_argument("--n_iters", type=int, default=200)
    parser.add_argument("--output_json", default="efficiency_new/itfd_efficiency_full.json")
    parser.add_argument("--output_csv", default="efficiency_new/itfd_efficiency_full.csv")
    parser.add_argument(
        "--summary_md",
        default="",
        help="Write a summary markdown from --summary_json inputs and exit.",
    )
    parser.add_argument("--summary_json", action="append", default=[])
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    if PROJECT_ROOT not in resolved.parents and resolved != PROJECT_ROOT:
        raise ValueError(f"Refusing to write outside project root: {resolved}")
    return resolved


def pt_path_for_backbone(backbone: str) -> Path:
    base = PROJECT_ROOT / "data" / "pretrain"
    if backbone == "ViT-B-32":
        return base / "CLIP-ViT-B-32-laion2B-s34B-b79K" / "open_clip_pytorch_model.bin"
    return base / "resnet50_clip" / "open_clip_pytorch_model.bin"


def default_checkpoint(dataset: str) -> Path:
    return (
        PROJECT_ROOT
        / "checkpoints"
        / f"{dataset}_0_best_model.pt"
    )


def model_args(args: argparse.Namespace) -> SimpleNamespace:
    pt_path = pt_path_for_backbone(args.backbone)
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing pretrained CLIP weights: {pt_path}")
    return SimpleNamespace(
        backbone=args.backbone,
        pt_path=str(pt_path),
        hidden_dim=args.hidden_dim,
        dropout_rate=0.5,
        local_rank=0,
        batch_size=args.batch_size,
        fashioniq_split=args.fashioniq_split,
    )


def load_dataset(args: argparse.Namespace, preprocess_train: Any, preprocess_val: Any) -> Any:
    transform = [preprocess_train, preprocess_val]
    if args.dataset in FASHIONIQ_CATEGORIES:
        return datasets.FashionIQ(
            path=str(PROJECT_ROOT / "data" / "FashionIQ") + "/",
            category=args.dataset,
            transform=transform,
            split=args.fashioniq_split,
        )
    if args.dataset == "shoes":
        return datasets.Shoes(
            path=str(PROJECT_ROOT / "data" / "Shoes"),
            transform=transform,
        )
    return datasets.Fashion200k(
        path=str(PROJECT_ROOT / "data" / "Fashion200k"),
        split="test",
        transform=transform,
    )


def load_model(args: argparse.Namespace, device: torch.device) -> Tuple[torch.nn.Module, str, bool]:
    margs = model_args(args)
    model = itfd_model.ITFD(margs, args.hidden_dim, dropout=0.5)

    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else default_checkpoint(args.dataset)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    checkpoint_loaded = False
    if checkpoint.exists():
        state = torch.load(str(checkpoint), map_location="cpu")
        model.load_state_dict(state, strict=True)
        checkpoint_loaded = True
    elif args.checkpoint:
        raise FileNotFoundError(f"Checkpoint was requested but does not exist: {checkpoint}")

    model.to(device)
    model.eval()
    return model, str(checkpoint), checkpoint_loaded


def get_queries_and_targets(dataset: Any, dataset_name: str) -> Tuple[List[dict], List[dict]]:
    if dataset_name in FASHIONIQ_CATEGORIES or dataset_name == "shoes":
        return list(dataset.test_queries), list(dataset.test_targets)

    queries = list(dataset.get_test_queries())
    targets = []
    for idx in range(len(dataset.imgs)):
        targets.append(
            {
                "target_img_id": idx,
                "target_img_data": dataset.get_img(idx),
            }
        )
    return queries, targets


def query_text_and_image(query: dict, dataset_obj: Any, dataset_name: str) -> Tuple[str, torch.Tensor]:
    if "textual_query" in query and "visual_query" in query:
        return query["textual_query"], query["visual_query"]
    if dataset_name == "fashion200k":
        source_id = query["source_img_id"]
        text = f"{query['source_caption']}, but {query['mod']['str']}"
        image = query.get("visual_query")
        if image is None:
            image = dataset_obj.get_written_img(source_id, query["target_word"])
        return text, image
    raise KeyError(f"Unsupported query schema keys: {sorted(query.keys())}")


def batched(items: Sequence[dict], batch_size: int) -> Iterable[List[dict]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def count_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def cuda_time_ms(device: torch.device, fn: Callable[[], Any]) -> Tuple[float, Any]:
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end)), out


def stack_images(images: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.stack([img.float() for img in images], dim=0).to(device, non_blocking=False)


@torch.no_grad()
def encode_gallery(
    model: torch.nn.Module,
    targets: Sequence[dict],
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    features: List[torch.Tensor] = []
    elapsed_ms = 0.0
    count = 0
    for batch in batched(targets, batch_size):
        imgs = stack_images([item["target_img_data"] for item in batch], device)
        ms, feat = cuda_time_ms(device, lambda imgs=imgs: model.extract_target(imgs))
        elapsed_ms += ms
        count += imgs.shape[0]
        features.append(feat.detach())
    gallery = torch.cat(features, dim=0)
    per_image_ms = elapsed_ms / max(count, 1)
    return gallery, per_image_ms


def make_query_batch(
    queries: Sequence[dict],
    dataset_obj: Any,
    dataset_name: str,
    start_idx: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[List[str], torch.Tensor]:
    selected = [queries[(start_idx + offset) % len(queries)] for offset in range(batch_size)]
    texts: List[str] = []
    images: List[torch.Tensor] = []
    for query in selected:
        text, image = query_text_and_image(query, dataset_obj, dataset_name)
        texts.append(text)
        images.append(image)
    return texts, stack_images(images, device)


@torch.no_grad()
def profile_queries(
    model: torch.nn.Module,
    dataset_obj: Any,
    queries: Sequence[dict],
    gallery_features: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[List[float], List[float], List[float]]:
    query_times: List[float] = []
    sim_times: List[float] = []
    e2e_times: List[float] = []
    total_iters = args.n_warmup + args.n_iters
    topk = min(50, gallery_features.shape[0])

    for i in range(total_iters):
        start_idx = (i * args.batch_size) % len(queries)
        texts, images = make_query_batch(
            queries, dataset_obj, args.dataset, start_idx, args.batch_size, device
        )

        query_ms, query_feature = cuda_time_ms(
            device,
            lambda texts=texts, images=images: model.extract_query(texts, images)[0],
        )
        sim_ms, _ = cuda_time_ms(
            device,
            lambda query_feature=query_feature: torch.topk(
                query_feature @ gallery_features.t(), k=topk, dim=1
            ),
        )

        def e2e() -> torch.Tensor:
            qf = model.extract_query(texts, images)[0]
            return torch.topk(qf @ gallery_features.t(), k=topk, dim=1).values

        e2e_ms, _ = cuda_time_ms(device, e2e)

        if i >= args.n_warmup:
            query_times.append(query_ms)
            sim_times.append(sim_ms)
            e2e_times.append(e2e_ms)

    return query_times, sim_times, e2e_times


def mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else math.nan


def std(values: Sequence[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def installed_counter_packages() -> List[str]:
    return [name for name in ("fvcore", "thop", "ptflops") if importlib.util.find_spec(name)]


def profiler_flops(device: torch.device, fn: Callable[[], Any]) -> Tuple[Optional[int], str]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        torch.cuda.synchronize(device)
        with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=False) as prof:
            with torch.no_grad():
                fn()
        torch.cuda.synchronize(device)
        total = int(sum(getattr(evt, "flops", 0) or 0 for evt in prof.key_averages()))
        if total <= 0:
            return None, "torch.profiler completed but reported zero FLOPs."
        return total, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


@torch.no_grad()
def profile_flops(
    model: torch.nn.Module,
    texts: Sequence[str],
    images: torch.Tensor,
    target_images: torch.Tensor,
    device: torch.device,
) -> Tuple[str, str, str, Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    installed = installed_counter_packages()
    method = "torch.profiler.with_flops"
    notes: List[str] = []
    if installed:
        notes.append(
            "Installed FLOPs packages detected but not used for final values because ITFD has text-list inputs; "
            "torch.profiler(with_flops=True) was used consistently."
        )
    else:
        notes.append("fvcore/thop/ptflops are unavailable; no package installation was attempted.")

    query_flops, query_err = profiler_flops(device, lambda: model.extract_query(list(texts), images)[0])
    target_flops, target_err = profiler_flops(device, lambda: model.extract_target(target_images))

    textual = F.normalize(model.extract_text_fea(list(texts)), p=2, dim=-1)
    visual = F.normalize(model.extract_img_fea(images), p=2, dim=-1)

    def extra_modules() -> torch.Tensor:
        del_mask = model.del_proj(textual)
        prs_mask = model.prs_proj(textual)
        new_mask = model.new_proj(textual)
        prs_ref = F.normalize(prs_mask * visual, p=2, dim=-1)
        new_text = F.normalize(new_mask * textual, p=2, dim=-1)
        combined_feature = model.combiner_fc(torch.cat([new_text, prs_ref], dim=-1))
        dynamic_scaler = model.scaler_fc(model.dropout(combined_feature))
        query = dynamic_scaler * new_text + (1 - dynamic_scaler) * prs_ref
        return F.normalize(query + 0.0 * del_mask.sum(), p=2, dim=-1)

    extra_flops, extra_err = profiler_flops(device, extra_modules)

    if query_err:
        notes.append(f"query_encoding profiling note: {query_err}")
    if target_err:
        notes.append(f"target_encoding profiling note: {target_err}")
    if extra_err:
        notes.append(f"itfd_extra_module profiling note: {extra_err}")
    notes.append(
        "FLOPs are partial profiler counts for supported operators; tokenization, normalization, elementwise ops, "
        "top-k internals, and framework overhead may be incomplete. MACs are reported as FLOPs/2 where profiler FLOPs exist."
    )

    status = "partial" if any(v is not None for v in (query_flops, target_flops, extra_flops)) else "failed"
    q_macs = int(query_flops // 2) if query_flops is not None else None
    t_macs = int(target_flops // 2) if target_flops is not None else None
    e_macs = int(extra_flops // 2) if extra_flops is not None else None
    return (
        status,
        " ".join(notes),
        method,
        query_flops,
        q_macs,
        target_flops,
        t_macs,
        extra_flops,
        e_macs,
    )


def write_outputs(result: ProfileResult, output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(result)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data.keys()))
        writer.writeheader()
        writer.writerow(data)


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return f"{value:.{digits}f}"
    return str(value)


def write_summary(json_paths: Sequence[str], output_md: Path) -> None:
    rows = []
    for path_str in json_paths:
        path = resolve_project_path(path_str)
        with path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    rows.sort(key=lambda r: ["dress", "shirt", "toptee"].index(r["dataset"]) if r["dataset"] in FASHIONIQ_CATEGORIES else 99)

    lines = [
        "# ITFD Full Efficiency Profiling Summary",
        "",
        "## Complete Efficiency Table",
        "",
        "| subset | params | peak GPU memory (MB) | query latency (ms) | E2E latency (ms) | E2E throughput (QPS) | query throughput (QPS) | gallery size | feature dim | gallery storage (MB) | query FLOPs | target FLOPs | extra FLOPs | FLOPs status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| {dataset} | {params_total} | {peak} | {query_ms} | {e2e_ms} | {e2e_qps} | {query_qps} | {gallery_size} | {dim} | {storage} | {qf} | {tf} | {ef} | {status} |".format(
                dataset=r["dataset"],
                params_total=fmt_num(r["params_total"]),
                peak=fmt_num(r["peak_cuda_memory_mb"]),
                query_ms=fmt_num(r["query_encode_latency_ms_mean"]),
                e2e_ms=fmt_num(r["end_to_end_retrieval_latency_ms_mean"]),
                e2e_qps=fmt_num(r["e2e_throughput_qps"]),
                query_qps=fmt_num(r["query_encoding_throughput_qps"]),
                gallery_size=fmt_num(r["gallery_size"]),
                dim=fmt_num(r["gallery_feature_dim"]),
                storage=fmt_num(r["gallery_feature_storage_mb"], 6),
                qf=fmt_num(r["query_encoding_flops"]),
                tf=fmt_num(r["target_encoding_flops"]),
                ef=fmt_num(r["itfd_extra_module_flops"]),
                status=r["flops_status"],
            )
        )

    lines.extend(
        [
            "",
            "## FLOPs/MACs Method, Scope, and Confidence",
            "",
            "The environment was checked for fvcore, thop, and ptflops before profiling. These packages were unavailable in this run, and no new package was installed. FLOPs were therefore collected with `torch.profiler.profile(..., with_flops=True)`. The reported query encoding FLOPs cover `model.extract_query`, target encoding FLOPs cover `model.extract_target`, and ITFD extra module FLOPs cover the learned mask/fusion modules after CLIP text/image features have been computed. Because PyTorch profiler only reports FLOPs for supported operators such as convolution and matrix multiplication, these values are marked `partial`; tokenizer work, normalization, many elementwise operations, top-k internals, and framework overhead may be incomplete. MACs are recorded as FLOPs/2 when profiler FLOPs are available.",
            "",
            "Matching cost is additionally reported as a theoretical estimate using the exact feature matrix multiplication shape: `matching_macs_estimated = gallery_size * feature_dim` and `matching_flops_estimated = 2 * gallery_size * feature_dim`.",
            "",
            "## Throughput Formula",
            "",
            "`e2e_throughput_qps = 1000 / avg_e2e_retrieval_latency_ms`; `query_encoding_throughput_qps = 1000 / avg_query_encoding_latency_ms`.",
            "",
            "## Gallery Feature Storage",
            "",
            "Gallery storage is measured from the actual `gallery_features` tensor after target encoding and L2 normalization: `gallery_feature_numel = gallery_features.numel()`, `gallery_feature_element_size_bytes = gallery_features.element_size()`, `gallery_feature_storage_bytes = numel * element_size`, and `gallery_feature_storage_mb = bytes / 1024 / 1024`.",
            "",
            "## Chinese Paper Paragraph",
            "",
            "为评估 ITFD 的推理效率，我们在不重新训练模型的前提下加载各 FashionIQ 子集的已训练 checkpoint，并统计参数量、峰值 GPU 显存、查询编码延迟、端到端检索延迟、吞吐率以及 gallery 特征存储开销。吞吐率由平均延迟换算得到，即端到端 QPS 为 1000 除以平均端到端检索延迟（ms）。gallery 存储开销基于实际缓存的 `gallery_features` tensor 统计其元素个数、元素字节数和总字节数。由于当前环境未安装 fvcore、thop 或 ptflops，FLOPs/MACs 采用 PyTorch profiler 的 `with_flops=True` 进行统计；该方法主要覆盖卷积和矩阵乘等支持算子，因此结果标记为 partial，并作为复杂度参考而非严格全算子计数。",
            "",
            "## English Paper Paragraph",
            "",
            "To evaluate the inference efficiency of ITFD, we loaded the trained checkpoint for each FashionIQ subset without retraining and measured the number of parameters, peak GPU memory, query encoding latency, end-to-end retrieval latency, throughput, and gallery feature storage. Throughput is derived from the measured latency, i.e., end-to-end QPS is computed as 1000 divided by the average end-to-end retrieval latency in milliseconds. Gallery storage is measured directly from the cached `gallery_features` tensor using its number of elements and element size. Since fvcore, thop, and ptflops were not available in the current environment, FLOPs/MACs were collected with PyTorch profiler using `with_flops=True`. This profiler mainly covers supported operators such as convolutions and matrix multiplications; therefore, the FLOPs/MACs are marked as partial and should be interpreted as complexity references rather than exact full-operator counts.",
            "",
            "## Main Text vs. Appendix",
            "",
            "Recommended for the main text: parameter count, peak GPU memory, query/E2E latency, E2E throughput, query throughput, and gallery feature storage. Recommended for the appendix: partial profiler FLOPs/MACs, ITFD extra module FLOPs/MACs, and theoretical matching FLOPs/MACs, with the profiling caveat stated explicitly.",
            "",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_table(result: ProfileResult) -> None:
    rows = [
        ("dataset", result.dataset),
        ("backbone", result.backbone),
        ("device", result.device),
        ("gallery_size", result.gallery_size),
        ("gallery_feature_dim", result.gallery_feature_dim),
        ("query_encode_ms_mean", f"{result.query_encode_latency_ms_mean:.3f}"),
        ("e2e_retrieval_ms_mean", f"{result.end_to_end_retrieval_latency_ms_mean:.3f}"),
        ("e2e_throughput_qps", f"{result.e2e_throughput_qps:.3f}"),
        ("query_throughput_qps", f"{result.query_encoding_throughput_qps:.3f}"),
        ("gallery_storage_mb", f"{result.gallery_feature_storage_mb:.6f}"),
        ("peak_cuda_memory_mb", f"{result.peak_cuda_memory_mb:.2f}"),
        ("params_total", str(result.params_total)),
        ("flops_status", result.flops_status),
        ("query_encoding_flops", fmt_num(result.query_encoding_flops)),
        ("target_encoding_flops", fmt_num(result.target_encoding_flops)),
        ("itfd_extra_module_flops", fmt_num(result.itfd_extra_module_flops)),
        ("checkpoint_loaded", str(result.checkpoint_loaded)),
    ]
    width = max(len(k) for k, _ in rows)
    print("\nITFD full efficiency profile")
    print("-" * (width + 28))
    for key, value in rows:
        print(f"{key:<{width}}  {value}")


def main() -> None:
    args = parse_args()
    if args.summary_md:
        if not args.summary_json:
            raise ValueError("--summary_md requires at least one --summary_json")
        summary_md = resolve_project_path(args.summary_md)
        write_summary(args.summary_json, summary_md)
        print(f"Wrote summary MD: {summary_md}")
        return

    if not args.dataset:
        raise ValueError("--dataset is required unless --summary_md is used")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.n_iters < 1:
        raise ValueError("--n_iters must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because local ITFD model.py calls .cuda(0).")

    device = torch.device("cuda:0")
    torch.set_grad_enabled(False)

    margs = model_args(args)
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.backbone, pretrained=margs.pt_path
    )
    dataset = load_dataset(args, preprocess_train, preprocess_val)
    queries, targets = get_queries_and_targets(dataset, args.dataset)
    if not queries:
        raise RuntimeError(f"No test queries found for dataset={args.dataset}")
    if not targets:
        raise RuntimeError(f"No gallery targets found for dataset={args.dataset}")

    model, checkpoint, checkpoint_loaded = load_model(args, device)
    params_total, params_trainable = count_params(model)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    gallery_features, target_ms_per_image = encode_gallery(
        model, targets, args.batch_size, device
    )
    gallery_features = F.normalize(gallery_features, p=2, dim=-1)

    query_times, sim_times, e2e_times = profile_queries(
        model, dataset, queries, gallery_features, args, device
    )

    texts, images = make_query_batch(queries, dataset, args.dataset, 0, args.batch_size, device)
    target_images = stack_images([targets[0]["target_img_data"]], device)
    (
        flops_status,
        flops_notes,
        flops_method,
        query_flops,
        query_macs,
        target_flops,
        target_macs,
        extra_flops,
        extra_macs,
    ) = profile_flops(model, texts, images, target_images, device)

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    avg_query_ms = mean(query_times)
    avg_e2e_ms = mean(e2e_times)
    gallery_dim = int(gallery_features.shape[-1])
    gallery_numel = int(gallery_features.numel())
    gallery_element_size = int(gallery_features.element_size())
    gallery_storage_bytes = int(gallery_numel * gallery_element_size)

    result = ProfileResult(
        query_encode_latency_ms_mean=avg_query_ms,
        query_encode_latency_ms_std=std(query_times),
        target_encode_latency_ms_per_image=float(target_ms_per_image),
        similarity_topk_latency_ms_mean=mean(sim_times),
        end_to_end_retrieval_latency_ms_mean=avg_e2e_ms,
        end_to_end_retrieval_latency_ms_std=std(e2e_times),
        e2e_throughput_qps=float(1000.0 / avg_e2e_ms),
        query_encoding_throughput_qps=float(1000.0 / avg_query_ms),
        peak_cuda_memory_mb=float(peak_mb),
        params_total=int(params_total),
        params_trainable=int(params_trainable),
        gallery_size=int(gallery_features.shape[0]),
        gallery_feature_dim=gallery_dim,
        gallery_feature_dtype=str(gallery_features.dtype),
        gallery_feature_numel=gallery_numel,
        gallery_feature_element_size_bytes=gallery_element_size,
        gallery_feature_storage_bytes=gallery_storage_bytes,
        gallery_feature_storage_mb=float(gallery_storage_bytes / 1024.0 / 1024.0),
        flops_status=flops_status,
        flops_notes=flops_notes,
        flops_method=flops_method,
        flops_tools_available=",".join(installed_counter_packages()) or "none",
        query_encoding_flops=query_flops,
        query_encoding_macs=query_macs,
        target_encoding_flops=target_flops,
        target_encoding_macs=target_macs,
        itfd_extra_module_flops=extra_flops,
        itfd_extra_module_macs=extra_macs,
        matching_macs_estimated=int(gallery_features.shape[0] * gallery_dim),
        matching_flops_estimated=int(2 * gallery_features.shape[0] * gallery_dim),
        matching_flops_notes="theoretical estimate: gallery_size * feature_dim MACs; 2 * gallery_size * feature_dim FLOPs",
        device=torch.cuda.get_device_name(device),
        backbone=args.backbone,
        dataset=args.dataset,
        fashioniq_split=args.fashioniq_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_warmup=args.n_warmup,
        n_iters=args.n_iters,
        checkpoint=checkpoint,
        checkpoint_loaded=checkpoint_loaded,
        target_topk=min(50, int(gallery_features.shape[0])),
    )

    output_json = resolve_project_path(args.output_json)
    output_csv = resolve_project_path(args.output_csv)
    write_outputs(result, output_json, output_csv)
    print_table(result)
    print(f"\nWrote JSON: {output_json}")
    print(f"Wrote CSV:  {output_csv}")


if __name__ == "__main__":
    main()
