"""Shared helpers for running the official Keras MCN and VLT models on the OptSAR-RSVG test split."""
import json
import os
import re

import numpy as np


def load_expressions(ann_file):
    """[(expression_id, image file relative to images/, caption)] from data/annotations/optsar_test.json."""
    data = json.load(open(ann_file))
    return [(im["id"], im["file_name"], im["caption"]) for im in data["images"]]


def normalize(caption):
    """RefCOCO sentences (what MCN/VLT were trained on) are lower-case words without punctuation."""
    return re.sub(r"[^a-z0-9 ]+", " ", caption.lower()).split()


def letterbox(path, size, style):
    """The models' own resize-and-pad. style='pil' (MCN, grey 128) or 'cv2' (VLT, grey 0.5)."""
    if style == "pil":
        from PIL import Image
        image = Image.open(path).convert("RGB")
        iw, ih = image.size
        scale = min(size / iw, size / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        dx, dy = (size - nw) // 2, (size - nh) // 2
        canvas = Image.new("RGB", (size, size), (128, 128, 128))
        canvas.paste(image.resize((nw, nh), Image.BICUBIC), (dx, dy))
        data = np.array(canvas) / 255.
    else:
        import cv2
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        ih, iw = image.shape[:2]
        scale = min(size / iw, size / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        dx, dy = (size - nw) // 2, (size - nh) // 2
        data = np.full((size, size, 3), 0.5)
        data[dy:dy + nh, dx:dx + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC) / 255.
    return data, (scale, dx, dy, iw, ih)


def to_original(boxes, meta):
    """Input-pixel xyxy boxes -> original image pixels, clipped to the image."""
    scale, dx, dy, iw, ih = meta
    boxes = (np.asarray(boxes, dtype=np.float64).reshape(-1, 4) - [dx, dy, dx, dy]) / scale
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, iw)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, ih)
    return boxes


def save(predictions, out_file):
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(predictions, f)
    print("saved {} predictions to {}".format(len(predictions), out_file))
