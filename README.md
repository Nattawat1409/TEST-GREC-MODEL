# GREC models on OptSAR-RSVG

This repo runs the four models from the GREC paper ([He et al., 2023](https://arxiv.org/abs/2308.16182)): **MDETR, UNINEXT, VLT and MCN**. Instead of gRefCOCO, they run on the **test split of [OptSAR-RSVG](https://huggingface.co/datasets/JunDong-dev/OptSAR-RSVG)**, a remote-sensing grounding dataset of optical and SAR images.

Nothing is trained here. The models stay exactly as released; only the dataset changes. Results are reported with the five metrics OptSAR-RSVG uses:

| Model | Pr@0.5 | Pr@0.7 | Pr@0.9 | mean IoU | cum IoU |
|---|---:|---:|---:|---:|---:|

## Before you start

You need:

- A Linux machine with an NVIDIA GPU and a working driver
- `conda` (Miniconda is fine)
- Docker with the NVIDIA Container Toolkit, for MCN and VLT only (they are TensorFlow 1 code, and Docker is the easiest way to get TF 1 on a modern GPU)
- About 40 GB of free disk space (most of it is the TensorFlow 1 Docker image)

Only MDETR runs on a Mac, on the CPU (see [MDETR on a Mac](#mdetr-on-a-mac-cpu-only)). UNINEXT needs a CUDA-only op, and MCN and VLT need the NVIDIA Docker setup.

```bash
git clone <this repo> TEST-GREC-MODEL
cd TEST-GREC-MODEL
```

## Step 1: Get the test data

This downloads only the test split: 4,434 images (about 1.6 GB) and 8,103 expressions. Train and val are not needed.

```bash
pip install -r requirements.txt
python scripts/download_optsar.py
python scripts/prepare_optsar_test.py
```

The second script writes `data/annotations/optsar_test.json`, with one entry per expression. All four models read this same file.

## Step 2: Run the models

Each model has a `setup.sh` (run once) and a `run.sh`. Every run writes a file like `outputs/mdetr_preds.json`. You can run them in any order, and you can skip any you don't need.

Some checkpoints are on OneDrive/SharePoint and can't be fetched by a script. When one is missing, `setup.sh` stops and prints the link and the file name to save it as.

### MDETR (GREC checkpoint from the GREC authors)

```bash
conda create -n grec-mdetr python=3.8 -y
conda activate grec-mdetr
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
bash models/mdetr/setup.sh     # downloads the GREC checkpoint automatically
bash models/mdetr/run.sh
```

#### MDETR on a Mac (CPU only)

MDETR has no custom CUDA code, so it also runs on a Mac (Apple Silicon or Intel) on the CPU. Two things differ from the Linux steps: the `+cu113` torch wheels don't exist for macOS, so install plain torch, and `run.sh` needs `DEVICE=cpu`.

```bash
conda create -n grec-mdetr python=3.8 -y
conda activate grec-mdetr
pip install torch torchvision
bash models/mdetr/setup.sh
DEVICE=cpu bash models/mdetr/run.sh
```

The full test split takes about 2 hours on an M2 Pro. Keep the machine awake while it runs. `outputs/mdetr_preds.json` is only written at the end, so `scripts/evaluate.py` finds nothing until the run finishes.

### UNINEXT (ResNet-50)

1. Download `model_final.pth` from the [UNINEXT image_joint_r50 folder](https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/EkfdtpEnPbZEjUToUGfJ_GMBXPRAPro27hc-tk40PUD8VA?e=8oIKkr) and save it as `checkpoints/uninext_image_joint_r50.pth`.
2. Then:

```bash
conda create -n grec-uninext python=3.8 -y
conda activate grec-uninext
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
conda install -y -c "nvidia/label/cuda-11.3.1" cuda-toolkit   # nvcc, used to build one CUDA op
bash models/uninext/setup.sh
bash models/uninext/run.sh     # NGPU=4 bash models/uninext/run.sh to use 4 GPUs
```

### MCN (DarkNet-53 + GRU)

1. Download the RefCOCO **Darknet** checkpoint linked in the [MCN README](https://github.com/luogen1996/MCN#pre-trained-models-and-logs) and save it as `checkpoints/mcn_refcoco.h5`.
2. Then:

```bash
bash models/mcn/setup.sh       # builds the grec-mcn Docker image (first time takes a while)
bash models/mcn/run.sh
```

### VLT (DarkNet-53 + GRU + transformer)

1. Download the RefCOCO model **and its .yaml config** from the [VLT model folder](https://entuedu-my.sharepoint.com/:f:/g/personal/liuc0058_e_ntu_edu_sg/EpE88e5DW1NEl6p7sKlMvrcBhBLeMTuHbtNKDiJCvhQBtQ?e=6thFDa). Put both files in `checkpoints/vlt_refcoco/`.
2. Then:

```bash
bash models/vlt/setup.sh       # builds the grec-vlt Docker image
bash models/vlt/run.sh
```

## Step 3: Make the table

```bash
python scripts/evaluate.py
```

This prints the 5-column table for every model that has a predictions file and saves it to `outputs/results.md`. Two useful flags:

- `--by-domain` adds separate optical and SAR rows.
- `--thresh 0.7` uses GREC's way of picking boxes: only boxes scoring 0.7 or more count, as in the paper's Table 3. The default uses each model's best box.

The metric code matches the OptSAR-RSVG authors' own (`scripts/metrics.py` in [OptiSAR-Net++](https://github.com/JunDong-dev/OptiSAR-Net-PlusPlus)). I checked both on the same predictions and they give identical numbers. An expression with no predicted box counts as IoU 0.

## What was changed in each model, and what to keep in mind

The official repos are cloned at fixed commits into `third_party/`, and the changes are kept as small as possible:

- **MDETR**: a ~30-line patch (`models/mdetr/optsar.patch`). It points the official GREC code at our annotation file, saves every box with its score, and stops the GREC evaluator from crashing on datasets with no "no-target" expressions (OptSAR has none). The checkpoint is the GREC-trained one released with the paper.
- **UNINEXT**: a ~50-line patch (`models/uninext/optsar.patch`). It registers the test split and keeps up to 100 boxes per expression instead of 1, which is the GREC change described in the paper. It also saves the boxes.
- **MCN**: no changes to its code. `models/mcn/infer.py` builds MCN with its own functions and config, and saves every box that survives NMS.
- **VLT**: no changes to its code. VLT only predicts a mask, so `models/vlt/infer.py` turns each connected region of the mask (thresholded at VLT's own 0.35) into a box.

Things that affect the numbers:

- **Only MDETR has a GREC-trained checkpoint.** The GREC authors didn't release theirs for UNINEXT, VLT or MCN, so those use each project's official RefCOCO-trained model. All four models are trained on natural photos and have never seen satellite or SAR images, so expect low scores. This is a zero-shot test.
- **MCN and VLT read at most 15 words**, as they were trained. About 5% of OptSAR test sentences are longer and get cut.
- **VLT predicts its mask at half resolution.** Even a perfect mask gives a box IoU of roughly 0.90–0.98, which mostly affects Pr@0.9.
- Captions are lower-cased and stripped of punctuation for MCN and VLT, to match the RefCOCO text they were trained on. MDETR and UNINEXT get the caption as written.

## Repo layout

```
scripts/download_optsar.py      get the test split from Hugging Face
scripts/prepare_optsar_test.py  build data/annotations/optsar_test.json
scripts/evaluate.py             Pr@0.5 / Pr@0.7 / Pr@0.9 / mean IoU / cum IoU table
models/<model>/setup.sh         clone the official code, patch it, install
models/<model>/run.sh           run on the test split, write outputs/<model>_preds.json
models/keras.Dockerfile         TensorFlow 1.15 image for MCN and VLT
```
