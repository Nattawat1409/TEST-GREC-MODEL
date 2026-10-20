#!/usr/bin/env bash
# Run VLT on the OptSAR-RSVG test split -> outputs/vlt_preds.json
set -euo pipefail
source "$(dirname "$0")/../common.sh"
check_data

CKPT_DIR="$ROOT/checkpoints/vlt_refcoco"
WEIGHTS="$(ls "$CKPT_DIR"/*.h5 | head -1)"
CONFIG="$(ls "$CKPT_DIR"/*.yaml | head -1)"

docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$ROOT":/work -w /work/third_party/VLT grec-vlt \
  python /work/models/vlt/infer.py \
    --config "/work/checkpoints/vlt_refcoco/$(basename "$CONFIG")" \
    --weights "/work/checkpoints/vlt_refcoco/$(basename "$WEIGHTS")" \
    --anns /work/data/annotations/optsar_test.json \
    --images /work/data/OptSAR-RSVG/images \
    --out /work/outputs/vlt_preds.json
