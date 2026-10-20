#!/usr/bin/env bash
# Run UNINEXT-R50 on the OptSAR-RSVG test split -> outputs/uninext_preds.json
# NGPU=4 bash models/uninext/run.sh   to use more GPUs (default 1)
set -euo pipefail
source "$(dirname "$0")/../common.sh"
check_data

cd "$ROOT/third_party/UNINEXT"
OPTSAR_PRED_FILE="$ROOT/outputs/uninext_preds.json" \
python3 launch.py --nn 1 --np "${NGPU:-1}" --eval-only --uni 1 \
  --config-file projects/UNINEXT/configs/eval_optsar_r50.yaml \
  MODEL.WEIGHTS "$ROOT/checkpoints/uninext_image_joint_r50.pth" \
  OUTPUT_DIR "$ROOT/outputs/uninext_run"
