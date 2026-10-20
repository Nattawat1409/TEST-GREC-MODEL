"""Turn OptSAR-RSVG test.json into one file that every model in this repo reads: data/annotations/optsar_test.json.

The dataset itself is not changed: every (image, expression, box) triple of the test split becomes one entry.
The layout is the per-expression COCO format used by the GREC MDETR code (and by UNINEXT for RefCOCO):

  images[i]       one per expression: file_name (relative to images/), width, height, id (= expression id),
                  caption, expressions [caption] (UNINEXT), domain (optical / sar), original_id
  annotations[i]  its single target box (COCO xywh), category_id 1, box polygon as segmentation,
                  tokens_positive (MDETR: the character span of the category name inside the caption)
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def category_span(caption, category_name):
    """Span MDETR aligns the box to: the category name, else its last word, else the whole caption."""
    low = caption.lower()
    for needle in (category_name.lower(), category_name.split()[-1].lower()):
        i = low.find(needle)
        if i >= 0:
            return [i, i + len(needle)]
    return [0, len(caption)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(ROOT / "data" / "OptSAR-RSVG"))
    p.add_argument("--out", default=str(ROOT / "data" / "annotations" / "optsar_test.json"))
    args = p.parse_args()

    root = Path(args.data_root)
    src = json.load(open(root / "test.json"))
    images = {im["id"]: im for im in src["images"]}
    missing = [im["file_name"] for im in src["images"] if not (root / "images" / "test" / im["file_name"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} test images are missing (e.g. {missing[0]}). Run scripts/download_optsar.py")

    out_images, out_anns = [], []
    for eid, a in enumerate(src["annotations"]):
        im = images[a["image_id"]]
        x, y, w, h = a["bbox"]
        out_images.append({
            "file_name": f"test/{im['file_name']}",
            "width": im["width"],
            "height": im["height"],
            "id": eid,
            "original_id": im["id"],
            "caption": a["caption"],
            "expressions": [a["caption"]],
            "dataset_name": "optsar",
            "tokens_negative": [],
            "domain": "sar" if im["file_name"].startswith("sar") else "optical",
        })
        out_anns.append({
            "id": eid,
            "image_id": eid,
            "category_id": 1,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
            "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
            "tokens_positive": [category_span(a["caption"], a["category_name"])],
            "original_id": a["id"],
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"images": out_images, "annotations": out_anns,
                   "categories": [{"id": 1, "name": "object", "supercategory": "object"}]}, f)
    n_sar = sum(im["domain"] == "sar" for im in out_images)
    print(f"wrote {args.out}: {len(out_images)} expressions ({len(out_images) - n_sar} optical, {n_sar} SAR) "
          f"on {len(images)} images")


if __name__ == "__main__":
    main()
