#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-dress}"
BACKBONE="${BACKBONE:-ViT-B-32}"
SPLIT="${SPLIT:-val-split}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-6}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

if [ "$BACKBONE" = "ViT-H-14" ]; then
  HIDDEN_DIM="${HIDDEN_DIM:-1024}"
else
  HIDDEN_DIM="${HIDDEN_DIM:-512}"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/outputs/$DATASET/$BACKBONE}"

python "$ROOT_DIR/src/train.py" \
  --dataset "$DATASET" \
  --fashioniq_split "$SPLIT" \
  --backbone "$BACKBONE" \
  --batch_size "$BATCH_SIZE" \
  --hidden_dim "$HIDDEN_DIM" \
  --num_workers "$NUM_WORKERS" \
  --model_dir "$MODEL_DIR"
