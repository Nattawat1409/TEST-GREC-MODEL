#!/usr/bin/env bash
# VLT: official Keras code + its RefCOCO checkpoint and config, run in a TF 1.15 Docker image.
set -euo pipefail
source "$(dirname "$0")/../common.sh"

need_file "$ROOT/checkpoints/vlt_refcoco" \
  "https://entuedu-my.sharepoint.com/:f:/g/personal/liuc0058_e_ntu_edu_sg/EpE88e5DW1NEl6p7sKlMvrcBhBLeMTuHbtNKDiJCvhQBtQ?e=6thFDa (put the RefCOCO .h5 and its .yaml in checkpoints/vlt_refcoco/)"

clone_at https://github.com/henghuiding/Vision-Language-Transformer.git "$ROOT/third_party/VLT" 9b24015566fa820e3eddbbd8942fa44512ec1b3c
docker build -f "$ROOT/models/keras.Dockerfile" --build-arg KERAS_VERSION=2.3.1 -t grec-vlt "$ROOT/models"
echo "VLT ready."
