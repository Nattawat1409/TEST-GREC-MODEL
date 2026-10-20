"""Run the official VLT (Keras, DarkNet-53 + bi-GRU + transformer) on the OptSAR-RSVG test split.

Run from inside the VLT repo (models/vlt/run.sh does this). VLT only predicts a segmentation mask, so boxes come
from the mask: it is thresholded with VLT's own segment_thresh, every connected region becomes one box, and its
score is the mean mask probability inside the region. That gives 0..N boxes per expression, the GREC way.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.getcwd())  # VLT repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # models/keras_common.py

import spacy  # noqa: E402
from keras.layers import Input  # noqa: E402
from yacs.config import CfgNode  # noqa: E402

import keras_common as kc  # noqa: E402
from loader.loader import qlist_to_vec  # noqa: E402
from model.vlt_model import yolo_body  # noqa: E402


def mask_to_boxes(prob, thresh):
    """Probability map (input size) -> xyxy boxes of its connected regions above thresh, with mean-prob scores."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((prob > thresh).astype(np.uint8), connectivity=8)
    boxes, scores = [], []
    for k in range(1, n):  # 0 is background
        x, y, w, h = stats[k, :4]
        boxes.append([x, y, x + w, y + h])
        scores.append(float(prob[labels == k].mean()))
    return np.array(boxes, dtype=np.float64).reshape(-1, 4), np.array(scores)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="config .yaml shipped with the VLT RefCOCO checkpoint")
    p.add_argument("--weights", required=True, help="VLT RefCOCO checkpoint (.h5)")
    p.add_argument("--anns", required=True, help="data/annotations/optsar_test.json")
    p.add_argument("--images", required=True, help="data/OptSAR-RSVG/images")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    with open("config/base.yaml") as f:
        config = CfgNode.load_cfg(f)
    config.merge_from_file(args.config)
    size = config.input_size

    model = yolo_body(Input(shape=(size, size, 3)), Input(shape=[config.word_len, config.embed_dim]), config)
    model.load_weights(args.weights, by_name=False, skip_mismatch=False)
    embed = spacy.load(config.word_embed)

    expressions = kc.load_expressions(args.anns)
    predictions = {}
    for start in range(0, len(expressions), args.batch_size):
        batch = expressions[start:start + args.batch_size]
        images, metas, words = [], [], []
        for _, file_name, caption in batch:
            image, meta = kc.letterbox(os.path.join(args.images, file_name), size, "cv2")
            images.append(image)
            metas.append(meta)
            words.append(qlist_to_vec(config.word_len, " ".join(kc.normalize(caption)), embed))
        masks = model.predict_on_batch([np.array(images), np.array(words)])
        for (eid, _, _), meta, logit in zip(batch, metas, masks):
            prob = cv2.resize(1. / (1. + np.exp(-logit[..., 0])), (size, size))
            boxes, scores = mask_to_boxes(prob, config.segment_thresh)
            predictions[str(eid)] = {"boxes": kc.to_original(boxes, meta).tolist(), "scores": scores.tolist()}
        print("\r{}/{}".format(len(predictions), len(expressions)), end="", flush=True)
    print()
    kc.save(predictions, args.out)


if __name__ == "__main__":
    main()
