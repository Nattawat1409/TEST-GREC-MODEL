"""Run the official MCN (Keras, DarkNet-53 + bi-GRU) on the OptSAR-RSVG test split.

Run from inside the MCN repo (models/mcn/run.sh does this). The model, its config (config/config.json) and the
YOLO box decoding are MCN's own. The only GREC change: instead of keeping the single best box, every box that
survives NMS is saved with its score (up to 100), so a score threshold can choose 0..N boxes.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())  # MCN repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # models/keras_common.py

import keras.backend as K  # noqa: E402
import spacy  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras.layers import Input  # noqa: E402

import keras_common as kc  # noqa: E402
from model.mcn_model import yolo_body, yolo_head  # noqa: E402
from utils.parse_config import config  # noqa: E402
from utils.utils import qlist_to_vec  # noqa: E402


def build_decoder(feat_shape, anchors, size, max_boxes, iou):
    """MCN's yolo_head on one image, then NMS. Returns xyxy boxes in input pixels and their scores."""
    feats = K.placeholder(shape=(1,) + tuple(feat_shape[1:]))
    box_xy, box_wh, conf = yolo_head(feats, anchors, K.constant([size, size]))
    xy, wh = box_xy * size, box_wh * size
    boxes = K.reshape(K.concatenate([xy - wh / 2, xy + wh / 2]), [-1, 4])
    scores = K.reshape(conf, [-1])
    yx = K.concatenate([boxes[:, 1:2], boxes[:, 0:1], boxes[:, 3:4], boxes[:, 2:3]])
    keep = tf.image.non_max_suppression(yx, scores, max_boxes, iou_threshold=iou)
    return feats, K.gather(boxes, keep), K.gather(scores, keep)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, help="MCN RefCOCO checkpoint (.h5)")
    p.add_argument("--anns", required=True, help="data/annotations/optsar_test.json")
    p.add_argument("--images", required=True, help="data/OptSAR-RSVG/images")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-boxes", type=int, default=100)
    p.add_argument("--nms-iou", type=float, default=0.5)
    args = p.parse_args()

    size = config["input_size"]
    anchors = np.array(open(config["anchors_file"]).readline().split(","), dtype=float).reshape(-1, 2)
    model = yolo_body(Input(shape=(size, size, 3)), Input(shape=[config["word_len"], config["embed_dim"]]),
                      len(anchors), config)
    model.load_weights(args.weights, by_name=True, skip_mismatch=True)
    feats_ph, boxes_t, scores_t = build_decoder(model.output_shape[0], anchors, size, args.max_boxes, args.nms_iou)
    sess = K.get_session()
    embed = spacy.load(config["word_embed"])

    expressions = kc.load_expressions(args.anns)
    predictions = {}
    for start in range(0, len(expressions), args.batch_size):
        batch = expressions[start:start + args.batch_size]
        images, metas, words = [], [], []
        for _, file_name, caption in batch:
            image, meta = kc.letterbox(os.path.join(args.images, file_name), size, "pil")
            images.append(image)
            metas.append(meta)
            words.append(qlist_to_vec(config["word_len"], kc.normalize(caption), embed))
        det_out = model.predict_on_batch([np.array(images), np.array(words)])[0]
        for (eid, _, _), meta, feat in zip(batch, metas, det_out):
            boxes, scores = sess.run([boxes_t, scores_t], {feats_ph: feat[None], K.learning_phase(): 0})
            predictions[str(eid)] = {"boxes": kc.to_original(boxes, meta).tolist(), "scores": scores.tolist()}
        print("\r{}/{}".format(len(predictions), len(expressions)), end="", flush=True)
    print()
    kc.save(predictions, args.out)


if __name__ == "__main__":
    main()
