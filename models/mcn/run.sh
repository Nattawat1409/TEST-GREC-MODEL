#!/usr/bin/env bash
# Run MCN on the OptSAR-RSVG test split -> outputs/mcn_preds.json
set -euo pipefail
source "$(dirname "$0")/../common.sh"
check_data

docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$ROOT":/work -w /work/third_party/MCN grec-mcn \
  python /work/models/mcn/infer.py \
    --weights /work/checkpoints/mcn_refcoco.h5 \
    --anns /work/data/annotations/optsar_test.json \
    --images /work/data/OptSAR-RSVG/images \
    --out /work/outputs/mcn_preds.json
