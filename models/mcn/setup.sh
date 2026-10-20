#!/usr/bin/env bash
# MCN: official Keras code + its RefCOCO (DarkNet-53) checkpoint, run in a TF 1.15 Docker image.
set -euo pipefail
source "$(dirname "$0")/../common.sh"

need_file "$ROOT/checkpoints/mcn_refcoco.h5" \
  "https://1drv.ms/u/s!AmrFUyZ_lDVGiRL_WITB7kfqX0St?e=JlWTBX (RefCOCO, Darknet; save the .h5 as checkpoints/mcn_refcoco.h5)"

clone_at https://github.com/luogen1996/MCN.git "$ROOT/third_party/MCN" f7c31897afd5ca5fe603aa35488d22ed3c5efa6b
docker build -f "$ROOT/models/keras.Dockerfile" --build-arg KERAS_VERSION=2.2.4 -t grec-mcn "$ROOT/models"
echo "MCN ready."
