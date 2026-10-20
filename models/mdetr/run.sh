#!/usr/bin/env bash
# Run MDETR (GREC checkpoint) on the OptSAR-RSVG test split -> outputs/mdetr_preds.json
#   GPU (default):  bash models/mdetr/run.sh
#   Mac / no GPU:   DEVICE=cpu bash models/mdetr/run.sh     (slow)
set -euo pipefail
source "$(dirname "$0")/../common.sh"
check_data

cd "$ROOT/third_party/gRefCOCO/mdetr"   # paths in configs/optsar.json are relative to this folder
python main.py --dataset_config configs/optsar.json \
  --resume "$ROOT/checkpoints/mdetr_grec.pth" --ema --eval \
  --batch_size "${BATCH:-4}" --num_workers 2 --device "${DEVICE:-cuda}"
