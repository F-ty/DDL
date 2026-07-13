#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-dress}"
CKPT="${2:-}"
BACKBONE="${BACKBONE:-ViT-B-32}"
SPLIT="${SPLIT:-val-split}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-6}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

if [ -z "$CKPT" ]; then
  echo "Usage: bash scripts/eval.sh <dataset> <checkpoint>"
  exit 1
fi

if [ "$BACKBONE" = "ViT-H-14" ]; then
  HIDDEN_DIM="${HIDDEN_DIM:-1024}"
else
  HIDDEN_DIM="${HIDDEN_DIM:-512}"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$ROOT_DIR/src/eval.py" \
  --dataset "$DATASET" \
  --fashioniq_split "$SPLIT" \
  --backbone "$BACKBONE" \
  --batch_size "$BATCH_SIZE" \
  --hidden_dim "$HIDDEN_DIM" \
  --num_workers "$NUM_WORKERS" \
  --ckpt "$CKPT"
