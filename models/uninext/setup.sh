#!/usr/bin/env bash
# UNINEXT-R50: official code + image_joint_r50 checkpoint (UNINEXT model zoo, manual download).
# Run inside the grec-uninext conda env (see README). Needs nvcc (CUDA toolkit) to build the deformable-attention op.
set -euo pipefail
source "$(dirname "$0")/../common.sh"
REPO="$ROOT/third_party/UNINEXT"

need_file "$ROOT/checkpoints/uninext_image_joint_r50.pth" \
  "https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/EkfdtpEnPbZEjUToUGfJ_GMBXPRAPro27hc-tk40PUD8VA?e=8oIKkr (model_final.pth, rename it)"

clone_at https://github.com/MasterBin-IIAU/UNINEXT.git "$REPO" 5ab03865d05d61c23799ae0bd3305b160ceb3f34
apply_patch "$REPO" "$ROOT/models/uninext/optsar.patch"

# data: UNINEXT looks for datasets under its own datasets/ folder
mkdir -p "$REPO/datasets"
ln -sfn "$ROOT/data" "$REPO/datasets/optsar"

# BERT-base text encoder, stored where UNINEXT expects it
BERT="$REPO/projects/UNINEXT/bert-base-uncased"
mkdir -p "$BERT"
for f in config.json vocab.txt pytorch_model.bin; do
  [ -f "$BERT/$f" ] || curl -L -o "$BERT/$f" "https://huggingface.co/bert-base-uncased/resolve/main/$f"
done

# install (from UNINEXT's INSTALL.md, with versions pinned so old code still runs)
cd "$REPO"
pip install -e .
pip install shapely==1.7.1 "pillow<10" "numpy<1.24" "transformers==4.25.1" \
  jpeg4py visdom easydict scikit-image tikzplotlib motmetrics \
  "git+https://github.com/XD7479/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI" \
  "git+https://github.com/lvis-dataset/lvis-api.git"
cd projects/UNINEXT/uninext/models/deformable_detr/ops && bash make.sh
echo "UNINEXT ready."
