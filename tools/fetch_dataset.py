"""Prepare the IDD data this project evaluates on."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "idd_lite"
MANIFEST = ROOT / "tools" / "idd_lite_manifest.json"


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest() -> dict:
    """Record what the reported results were measured on."""
    files = sorted(p for p in DATA.rglob("*") if p.is_file())
    entries = {str(p.relative_to(DATA)): digest(p) for p in files}
    return {
        "dataset": "IDD-Lite (Indian Driving Dataset), IIIT Hyderabad",
        "source": "https://idd.insaan.iiit.ac.in/",
        "note": "Registration required. Not redistributed with this project.",
        "file_count": len(entries),
        "val_frames": sum(1 for k in entries if "/val/" in k and k.endswith("_image.jpg")),
        "files": entries,
    }


def verify() -> int:
    if not MANIFEST.is_file():
        sys.exit("no manifest; run with --build-manifest on a known-good copy")
    m = json.loads(MANIFEST.read_text())
    missing = wrong = 0
    for rel, want in m["files"].items():
        p = DATA / rel
        if not p.is_file():
            missing += 1
        elif digest(p) != want:
            wrong += 1
    total = len(m["files"])
    print(f"  expected {total} files, {m['val_frames']} validation frames")
    print(f"  missing  {missing}")
    print(f"  mismatch {wrong}")
    if missing or wrong:
        print("\n  This copy differs from the one the reported results were "
              "measured on.")
        return 1
    print("\n  This copy matches the data used for every reported result.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", help="IDD-Lite archive downloaded after registering")
    ap.add_argument("--verify", action="store_true", help="check an existing copy")
    ap.add_argument("--build-manifest", action="store_true",
                    help="record checksums of the current copy")
    a = ap.parse_args()

    if a.build_manifest:
        MANIFEST.write_text(json.dumps(build_manifest(), indent=1))
        m = json.loads(MANIFEST.read_text())
        print(f"  manifest written: {m['file_count']} files, "
              f"{m['val_frames']} validation frames")
        return
    if a.archive:
        DATA.parent.mkdir(parents=True, exist_ok=True)
        print(f"  extracting {a.archive} ...")
        with tarfile.open(a.archive) as t:
            t.extractall(DATA.parent)
        print("  extracted; verifying")
    if a.archive or a.verify:
        sys.exit(verify())
    ap.print_help()


if __name__ == "__main__":
    main()
