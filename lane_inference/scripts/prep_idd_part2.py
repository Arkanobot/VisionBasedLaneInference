"""Prepare IDD Segmentation Part II locally and merge it into datasets/idd_work."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "datasets" / "_raw"
EXTRACT = RAW / "part2_extract"
CODE = RAW / "public-code"
WORK = PROJECT / "datasets" / "idd_work"


L3_TO_L1 = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2,
            6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3,
            13: 4, 14: 4, 15: 4, 16: 4, 17: 4, 18: 4, 19: 4, 20: 4, 21: 4,
            22: 5, 23: 5, 24: 5, 25: 6}


SPLIT_MAP = {"train": "train", "val": "heldout"}


def log(msg: str) -> None:
    print(msg, flush=True)


def extract(archive: Path) -> None:
    if EXTRACT.is_dir() and any(EXTRACT.iterdir()):
        log(f"extract: {EXTRACT} already populated, skipping")
        return
    EXTRACT.mkdir(parents=True, exist_ok=True)
    log(f"extract: {archive.name} ({archive.stat().st_size / 1e9:.1f} GB) -> {EXTRACT}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(EXTRACT)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(EXTRACT)
    else:
        head = archive.read_bytes()[:80]
        raise RuntimeError(f"{archive.name} is neither zip nor tar: {head!r}")


def find_roots(base: Path) -> list[Path]:
    """Every directory holding both leftImg8bit/ and gtFine/, not just the first."""
    roots = []
    for p in [base] + [d for d in base.rglob("*") if d.is_dir()]:
        if (p / "leftImg8bit").is_dir() and (p / "gtFine").is_dir():
            roots.append(p)

    return [r for r in roots if not any(r != o and o in r.parents for o in roots)]


def generate_labels(root: Path) -> str:
    if not CODE.is_dir():
        log("labels: cloning AutoNUE/public-code")
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/AutoNUE/public-code.git", str(CODE)],
                       check=True)


    for f in CODE.rglob("*.py"):
        t = f.read_text(errors="ignore")
        if "PILLOW_VERSION" in t:
            f.write_text(t.replace("PILLOW_VERSION", "__version__"))


    env = dict(os.environ, ANUE=str(root),
               PYTHONPATH=os.pathsep.join([str(CODE), str(CODE / "helpers"),
                                           str(CODE / "preperation")]))

    def run(id_type: str) -> int:
        existing = len(list((root / "gtFine").rglob(f"*label{id_type}s.png")))
        if existing:
            log(f"labels: {existing} {id_type} masks already present, skipping")
            return existing
        r = subprocess.run([sys.executable, "createLabels.py",
                            "--datadir", str(root), "--id-type", id_type,
                            "--num-workers", str(max(1, (os.cpu_count() or 4) // 2))],
                           cwd=str(CODE / "preperation"), env=env,
                           capture_output=True, text=True)
        made = len(list((root / "gtFine").rglob(f"*label{id_type}s.png")))
        log(f"labels: {id_type} -> rc {r.returncode}, {made} masks")
        if made == 0:
            log(((r.stdout or "") + (r.stderr or ""))[-1500:])
        return made

    if run("level1Id"):
        return "level1Id"
    log("labels: level1Id produced nothing, falling back to level3Id")
    if run("level3Id"):
        return "level3Id"
    raise RuntimeError("label generation failed for " + str(root))


def build_lut(level: str) -> np.ndarray:
    lut = np.full(256, 255, np.uint8)
    if level == "level3Id":
        for k, v in L3_TO_L1.items():
            lut[k] = v
    else:
        for k in range(7):
            lut[k] = k
    return lut


def merge(root: Path, level: str, size: tuple[int, int], workers: int) -> dict[str, int]:
    lut = build_lut(level)
    suffix = f"_gtFine_label{level}s.png"
    counts: dict[str, int] = {}

    def convert(args) -> int:
        img_path, dst_split = args
        stem = img_path.name.split("_leftImg8bit")[0]
        gtdir = Path(str(img_path.parent).replace("leftImg8bit", "gtFine"))
        lbl = gtdir / (stem + suffix)
        if not lbl.is_file():
            hits = list(gtdir.glob(stem + "*label*.png"))
            if not hits:
                return 0
            lbl = hits[0]
        im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        lb = cv2.imread(str(lbl), cv2.IMREAD_GRAYSCALE)
        if im is None or lb is None:
            return 0
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
        lb = lut[cv2.resize(lb, size, interpolation=cv2.INTER_NEAREST)]
        drive = f"p2_{img_path.parent.name}"
        di = WORK / "leftImg8bit" / dst_split / drive
        dl = WORK / "gtFine" / dst_split / drive
        di.mkdir(parents=True, exist_ok=True)
        dl.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(di / f"{stem}_image.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(dl / f"{stem}_label.png"), lb)
        return 1

    for src_split, dst_split in SPLIT_MAP.items():
        d = root / "leftImg8bit" / src_split
        if not d.is_dir():
            continue
        files = [(p, dst_split) for p in sorted(d.glob("*/*"))
                 if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        if not files:
            continue
        with ThreadPoolExecutor(workers) as ex:
            n = sum(ex.map(convert, files))
        log(f"merge: {src_split:5} -> {dst_split:8} {n}/{len(files)} converted")
        counts[dst_split] = counts.get(dst_split, 0) + n
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description="Merge IDD Part II into idd_work")
    p.add_argument("--archive", default=str(RAW / "part2.tar.gz"))
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--keep-extract", action="store_true",
                   help="keep the full-resolution tree (needed for a later "
                        "higher-resolution rebuild)")
    args = p.parse_args()

    if not args.skip_extract:
        extract(Path(args.archive))

    roots = find_roots(EXTRACT)
    if not roots:
        log(f"No leftImg8bit/+gtFine/ pair under {EXTRACT}. Tree (first 40):")
        for n, q in enumerate(sorted(EXTRACT.rglob("*"))):
            if n >= 40:
                break
            log("   " + str(q.relative_to(EXTRACT)))
        raise SystemExit(1)

    log(f"\nfound {len(roots)} dataset root(s):")
    for r in roots:
        polys = len(list((r / "gtFine").rglob("*_polygons.json")))
        per = {s: len(list((r / "leftImg8bit" / s).glob("*/*")))
               for s in ("train", "val", "test")
               if (r / "leftImg8bit" / s).is_dir()}
        log(f"   {r.relative_to(EXTRACT)}  polygons={polys}  {per}")

    total: dict[str, int] = {}
    for r in roots:
        level = generate_labels(r)
        for k, v in merge(r, level, (args.width, args.height), args.workers).items():
            total[k] = total.get(k, 0) + v

    log("\nmerged into datasets/idd_work:")
    for split in ("train", "val", "heldout"):
        d = WORK / "leftImg8bit" / split
        n = len(list(d.glob("*/*_image.jpg"))) if d.is_dir() else 0
        log(f"   {split:8} {n:6} images  (+{total.get(split, 0)} from Part II)")

    if not args.keep_extract:
        shutil.rmtree(EXTRACT, ignore_errors=True)
        log(f"\nremoved {EXTRACT}")
    log("\nNext: python -m scripts.make_lane_pseudolabels --splits train,heldout "
        "--skip-existing")


if __name__ == "__main__":
    main()
