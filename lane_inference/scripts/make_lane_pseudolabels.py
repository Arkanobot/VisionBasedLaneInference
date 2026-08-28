"""Generate lane-line pseudo-labels for IDD-Lite using YOLOP as a teacher."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import config as cfg
from data.idd import list_split


def main() -> None:
    p = argparse.ArgumentParser(description="Distil YOLOP lane-line labels")
    p.add_argument("--root", default=str(cfg.DATA_ROOT))
    p.add_argument("--splits", default="train,val")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true",
                   help="leave already-written pseudo-labels alone; used when\n                        adding a new dataset part to an existing set")
    args = p.parse_args()

    from core.dl_drivable import YolopBackend
    teacher = YolopBackend()
    root = Path(args.root)

    total, written, nonempty, skipped = 0, 0, 0, 0
    t0 = time.time()
    for split in args.splits.split(","):
        pairs = list_split(root, split.strip())
        if args.limit:
            pairs = pairs[:args.limit]
        for img_path, _ in pairs:
            total += 1
            drive = img_path.parent.name
            stem = img_path.name.replace("_image.jpg", "")
            out_dir = root / "laneFine" / split.strip() / drive
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}_lane.png"
            if args.skip_existing and out_path.is_file():
                skipped += 1
                continue

            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            _, lane, _ = teacher(image)
            cv2.imwrite(str(out_path), lane)
            written += 1
            if lane.any():
                nonempty += 1
            if written % 100 == 0:
                rate = written / max(time.time() - t0, 1e-6)
                print(f"  {written} written  ({rate:.1f}/s)", flush=True)

    dt = time.time() - t0
    print(f"\n{written}/{total} pseudo-labels written in {dt/60:.1f} min"
          + (f" ({skipped} already present, skipped)" if skipped else ""))
    print(f"{nonempty} ({100*nonempty/max(written,1):.1f}%) contain lane-line pixels")


if __name__ == "__main__":
    main()
