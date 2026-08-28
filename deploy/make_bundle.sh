#!/usr/bin/env bash
# Assemble a self-contained deployable copy under deploy/build/.
#
# The bundle mirrors the project's own directory layout, because every path in
# lane_inference/config.py resolves relative to the repository root -- so the
# application runs in the container with no path changes at all.
#
#   ./deploy/make_bundle.sh
#
# Nothing is built or run locally; this only copies files.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=deploy/build
SAMPLE_IMAGES=16     # /api/samples picks 8 evenly spaced from whatever is here
SAMPLE_CLIPS=6

rm -rf "$OUT"
mkdir -p "$OUT"

if [ ! -d web/dist ]; then
  echo "web/dist is missing. Run 'npm run build' in web/ first." >&2
  exit 1
fi

# --- application code -----------------------------------------------------
mkdir -p "$OUT/server" "$OUT/lane_inference" "$OUT/web"
rsync -a --exclude='__pycache__' --exclude='*.pyc' server/ "$OUT/server/"
# certs/ holds a self-signed key pair for serving HTTPS on the LAN. The host
# terminates TLS itself, and a private key has no business in a shared image.
rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='tests' \
         --exclude='.pytest_cache' --exclude='certs' \
         lane_inference/ "$OUT/lane_inference/"
rsync -a web/dist/ "$OUT/web/dist/"

# --- the shipped checkpoint ------------------------------------------------
# best.pt only. last.pt is optimiser state for resuming training, 40 MB of no
# use to a server that never trains.
mkdir -p "$OUT/outputs/checkpoints/seg_idd20k"
cp outputs/checkpoints/seg_idd20k/best.pt \
   outputs/checkpoints/seg_idd20k/config.json \
   outputs/checkpoints/seg_idd20k/history.json \
   "$OUT/outputs/checkpoints/seg_idd20k/"

# --- measured results, read live by /api/results ---------------------------
mkdir -p "$OUT/outputs/results"
cp outputs/results/*.md outputs/results/*.json "$OUT/outputs/results/"

# --- a sample of validation frames and clips -------------------------------
# The full IDD-Lite split is 46 MB and the clip set 160 MB; the demo only ever
# offers eight of each.
mkdir -p "$OUT/datasets/idd_lite/leftImg8bit/val" "$OUT/datasets/idd_clips"
find datasets/idd_lite/leftImg8bit/val -name '*_image.jpg' | sort |
  awk -v n="$SAMPLE_IMAGES" 'NR%3==1 && c<n {print; c++}' |
  while read -r f; do
    d="$OUT/${f#./}"; mkdir -p "$(dirname "$d")"; cp "$f" "$d"
  done
ls datasets/idd_clips/*.mp4 | sort | head -"$SAMPLE_CLIPS" |
  xargs -I{} cp {} "$OUT/datasets/idd_clips/"

# --- container recipe ------------------------------------------------------
cp deploy/Dockerfile deploy/requirements.txt "$OUT/"
cp deploy/space_README.md "$OUT/README.md"
cp deploy/dockerignore "$OUT/.dockerignore"

echo
echo "  bundle: $OUT  ($(du -sh "$OUT" | cut -f1))"
find "$OUT" -maxdepth 2 -mindepth 1 -type d | sed 's|^|    |' | sort
echo
echo "  images: $(find "$OUT/datasets/idd_lite" -name '*_image.jpg' | wc -l | tr -d ' ')  clips: $(ls "$OUT/datasets/idd_clips" | wc -l | tr -d ' ')"
