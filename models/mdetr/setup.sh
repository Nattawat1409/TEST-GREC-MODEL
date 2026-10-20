#!/usr/bin/env bash
# MDETR: official GREC code (gRefCOCO repo) + the GREC-trained checkpoint released by the GREC authors.
# Run inside the grec-mdetr conda env (see README).
set -euo pipefail
source "$(dirname "$0")/../common.sh"
REPO="$ROOT/third_party/gRefCOCO"

clone_at https://github.com/henghuiding/gRefCOCO.git "$REPO" 41a8f008006bb1fb0f1df2547f9477bb97e36593
apply_patch "$REPO" "$ROOT/models/mdetr/optsar.patch"
# the repo's requirements.txt pins transformers==4.5.1 (needs Python 3.8) and panopticapi (never imported), so
# install the packages evaluation actually uses instead; torch/torchvision must already be installed.
# transformers>=4.31 drops RoBERTa's position_ids buffer, so the GREC checkpoint no longer loads;
# torch/torchvision are pinned to what is installed so timm can't upgrade them.
TORCH_PINS=$(python -c 'import torch, torchvision; print(f"torch=={torch.__version__} torchvision=={torchvision.__version__}")')
pip install cython scipy xmltodict tqdm prettytable submitit timm onnx onnxruntime pycocotools \
  "transformers<4.31" sentencepiece gdown $TORCH_PINS

CKPT="$ROOT/checkpoints/mdetr_grec.pth"
[ -f "$CKPT" ] || gdown 14OrM3n_Oap7xCT6nxj9QEnkJOUMpBGjB -O "$CKPT"
echo "MDETR ready."
