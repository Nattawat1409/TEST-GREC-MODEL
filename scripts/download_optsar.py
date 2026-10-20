"""Download the OptSAR-RSVG test split (test.json + images/test: 4,434 images, about 1.6 GB) into data/OptSAR-RSVG.

Safe to run again: files that are already there are skipped. If Hugging Face answers "429 Too Many Requests",
the script waits and continues by itself.
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTP download: avoids the per-file token requests that trigger 429

from huggingface_hub import snapshot_download  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ROOT / "data" / "OptSAR-RSVG"))
    p.add_argument("--workers", type=int, default=4, help="parallel downloads (lower it if you still get 429)")
    args = p.parse_args()

    for attempt in range(1, 31):
        try:
            snapshot_download(
                "JunDong-dev/OptSAR-RSVG",
                repo_type="dataset",
                local_dir=args.out,
                allow_patterns=["README.md", "test.json", "images/test/*"],
                max_workers=args.workers,
            )
            break
        except Exception as e:  # rate limit or a dropped connection: wait, then continue where it stopped
            wait = min(30 * attempt, 300)
            print(f"\nattempt {attempt} stopped ({type(e).__name__}: {str(e)[:120]}). Retrying in {wait}s...")
            time.sleep(wait)
    else:
        raise SystemExit("Still failing after 30 attempts. Wait a few minutes and run the script again.")

    n = len(list(Path(args.out, "images", "test").glob("*.jpg")))
    print(f"done: {n} test images in {args.out}")
    if n != 4434:
        print("warning: expected 4434 images. Run the script again to fetch the missing ones.")


if __name__ == "__main__":
    main()
