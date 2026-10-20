"""Score model predictions on the OptSAR-RSVG test split: Pr@0.5, Pr@0.7, Pr@0.9, mean IoU, cum IoU.

The metrics follow OptiSAR-Net++ (scripts/metrics.py of the OptSAR-RSVG authors): one box per expression,
boxes clipped to the image, a missing box counts as IoU 0 (its target area still enters the cum IoU union).

Each model writes outputs/<model>_preds.json = {expression_id: {"boxes": [[x1, y1, x2, y2], ...], "scores": [...]}}
in original image pixels. The box used is the highest-scoring one. With --thresh T (the GREC way of choosing
boxes), only boxes scoring >= T count, so an expression with no box above T is a miss.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["mdetr", "uninext", "vlt", "mcn"]
COLUMNS = ["Pr@0.5", "Pr@0.7", "Pr@0.9", "mean IoU", "cum IoU"]


def intersection_union(pred, gt_xywh, width, height):
    x, y, w, h = gt_xywh
    tx1, ty1, tx2, ty2 = np.clip(x, 0, width), np.clip(y, 0, height), np.clip(x + w, 0, width), np.clip(y + h, 0, height)
    target_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    if pred is None:
        return 0.0, target_area
    px1, px2 = np.clip([pred[0], pred[2]], 0, width)
    py1, py2 = np.clip([pred[1], pred[3]], 0, height)
    inter = max(0.0, min(px2, tx2) - max(px1, tx1)) * max(0.0, min(py2, ty2) - max(py1, ty1))
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    return inter, pred_area + target_area - inter


def pick_box(pred, thresh):
    if not pred or not pred["scores"]:
        return None
    i = int(np.argmax(pred["scores"]))
    if thresh is not None and pred["scores"][i] < thresh:
        return None
    return pred["boxes"][i]


def score(gt, preds, thresh=None, domain=None):
    boxes = {a["image_id"]: a["bbox"] for a in gt["annotations"]}
    records = []
    for im in gt["images"]:
        if domain and im["domain"] != domain:
            continue
        box = pick_box(preds.get(str(im["id"])), thresh)
        records.append(intersection_union(box, boxes[im["id"]], im["width"], im["height"]))
    ious = np.array([i / u if u else 0.0 for i, u in records])
    return {
        "Pr@0.5": 100 * np.mean(ious >= 0.5),
        "Pr@0.7": 100 * np.mean(ious >= 0.7),
        "Pr@0.9": 100 * np.mean(ious >= 0.9),
        "mean IoU": 100 * ious.mean(),
        "cum IoU": 100 * sum(i for i, _ in records) / sum(u for _, u in records),
        "n": len(records),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", default=str(ROOT / "data" / "annotations" / "optsar_test.json"))
    p.add_argument("--outputs", default=str(ROOT / "outputs"))
    p.add_argument("--thresh", type=float, default=None, help="GREC score threshold (default: plain top-1 box)")
    p.add_argument("--by-domain", action="store_true", help="also report optical and SAR separately")
    args = p.parse_args()

    gt = json.load(open(args.gt))
    rows = []
    for model in MODELS:
        path = Path(args.outputs) / f"{model}_preds.json"
        if not path.exists():
            continue
        preds = json.load(open(path))
        missing = sum(str(im["id"]) not in preds for im in gt["images"])
        if missing:
            print(f"warning: {model} has no prediction for {missing} expressions (counted as misses)")
        for domain in ([None, "optical", "sar"] if args.by_domain else [None]):
            rows.append((model if domain is None else f"{model} ({domain})", score(gt, preds, args.thresh, domain)))

    if not rows:
        raise SystemExit(f"no predictions found in {args.outputs} (expected <model>_preds.json)")
    lines = ["| Model | " + " | ".join(COLUMNS) + " |", "|---|" + "---:|" * len(COLUMNS)]
    lines += [f"| {name} | " + " | ".join(f"{r[c]:.2f}" for c in COLUMNS) + " |" for name, r in rows]
    table = "\n".join(lines)
    note = "top-1 box" if args.thresh is None else f"boxes with score >= {args.thresh}"
    print(f"OptSAR-RSVG test, {rows[0][1]['n']} expressions, {note}\n\n{table}")
    out = Path(args.outputs) / "results.md"
    out.write_text(f"OptSAR-RSVG test ({note})\n\n{table}\n")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
