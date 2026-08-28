# Setup and Run Guide

Vision-Based Lane Inference for Unstructured & Structured Indian Roads
BITS Pilani · BSc Computer Science · Study Project, Phase 4

This document covers environment setup, data placement, and every command
needed to reproduce the reported results. For what the system does and why, see
`README.md`.

---

## 1. Requirements

| | |
|---|---|
| Python | 3.10 to 3.12 |
| RAM | 8 GB (16 GB to train locally) |
| Disk | ~2 GB including dependencies |
| GPU | Optional. Apple MPS and CUDA are both used automatically if present; CPU works. |

Verified on macOS 26.5 with an Apple M2 Pro (MPS) and on Google Colab with a
Tesla T4 (CUDA).

## 2. Environment

```bash
cd "ODP Project"
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r lane_inference/requirements.txt
```

Confirm the install and see which compute backend was selected:

```bash
cd lane_inference
PYTHONPATH=. python -c "
import config, torch
print('torch  :', torch.__version__)
print('device :', config.resolve_device())"
```

Expected: `mps` on Apple silicon, `cuda` on an NVIDIA machine, otherwise `cpu`.

### macOS certificate note

python.org builds of Python ship without a CA bundle, which makes `torch.hub`
and torchvision weight downloads fail with `CERTIFICATE_VERIFY_FAILED`.
`config.py` points OpenSSL at `certifi` on import, so this is handled
automatically, but only for code that imports `config` first.

## 3. Data

The project uses **IDD-Lite** (Indian Driving Dataset, level-1 label hierarchy):
1,403 train / 204 val / 404 test frames at 320×227, with 7-class ground truth.

Place it so the tree looks like this:

```
ODP Project/
  datasets/idd_lite/
    leftImg8bit/{train,val,test}/<drive>/<frame>_image.jpg
    gtFine/{train,val}/<drive>/<frame>_label.png
  lane_inference/
  outputs/
```

Verify:

```bash
cd lane_inference
PYTHONPATH=. python -c "
from data.idd import IDDLite
import config as cfg
for s in ('train','val'):
    print(s, len(IDDLite(cfg.DATA_ROOT, s)))"
```

Expected: `train 1403`, `val 204`.

## 4. Model weights

Put the trained checkpoint at:

```
outputs/checkpoints/seg_mnv3/best.pt
```

Either train it yourself (section 6) or take it from the Colab run
(`dist/train_colab.ipynb` writes `seg_mnv3.zip` to Drive; unzip so that
`checkpoints/seg_mnv3/best.pt` lands at the path above).

Without a checkpoint everything still runs, falling back to the classical
backend, and every learned configuration is skipped with a message.

## 5. Running

All commands are run from `lane_inference/` with the venv active.

```bash
# single image
python main.py image ../datasets/idd_lite/leftImg8bit/val/119/903127_image.jpg

# with the four-panel diagnostic view (input / lanes / semantics / rectified)
python main.py image road.jpg --panel

# save instead of displaying, e.g. over SSH
python main.py image road.jpg --panel --no-show --save out.png

# video; temporal smoothing turns on automatically
python main.py video drive.mp4 --panel --save annotated.mp4

# webcam
python main.py camera 0 --panel

# a grid of dataset frames, for report figures
python main.py dataset --split val --count 8 --save ../outputs/grid.png --no-show

# per-stage latency
python main.py benchmark --backend all
```

The Phase 3 invocation still works: `python main.py <image path>`.

### Options worth knowing

| Flag | Meaning |
|---|---|
| `--backend cv` | classical only, no model needed |
| `--backend dl` | learned model only |
| `--backend hybrid` | learned model with a classical fallback on collapse (default) |
| `--camera-height M` | the single metric calibration parameter, default 1.75 |
| `--max-range M` | how far ahead the overlay is drawn, default 22 m |
| `--blend` | ablation only: blend CV and DL confidence (measured to reduce IoU) |
| `--no-show` | headless |

## 6. Training

```bash
cd lane_inference
PYTHONPATH=. python -m models.train \
    --epochs 150 --batch-size 32 --lr 6e-4 --run-name seg_mnv3
```

Roughly 30 min on a Colab T4, 50 min on Apple MPS. Writes `best.pt`, `last.pt`,
`history.json` and `config.json` to `outputs/checkpoints/seg_mnv3/`.
`history.json` is rewritten every epoch, so progress can be inspected while
training runs.

For Colab, open `dist/train_colab.ipynb`, set the runtime to a T4 GPU, upload
`dist/odp_bundle.zip` to `MyDrive/ODP/`, and run all cells.

## 7. Reproducing the reported results

```bash
cd lane_inference

# full ablation, including the Phase 3 configurations and YOLOP
PYTHONPATH=. python -m eval.evaluate --split val

# faster, skipping the CPU-bound YOLOP runs
PYTHONPATH=. python -m eval.evaluate --split val --skip-yolop

# sensitivity of lane inference to the metric calibration
PYTHONPATH=. python -m eval.sensitivity
```

Results are written to `outputs/results/` as both Markdown and JSON.
The full evaluation takes about 4 minutes; YOLOP runs on CPU at ~6 FPS and
dominates that time.

## 8. Troubleshooting

**`ModuleNotFoundError: No module named 'config'`**
Run from inside `lane_inference/` with `PYTHONPATH=.` set.

**`CERTIFICATE_VERIFY_FAILED` on macOS**
Import `config` before `torch.hub`, or `pip install certifi` and export
`SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")`.

**`No module named 'prefetch_generator'` when loading YOLOP**
YOLOP's own dependency: `pip install prefetch_generator yacs`.

**`No checkpoint at .../best.pt`**
Expected without trained weights. See section 4, or pass `--backend cv`.

**Nothing appears when running over SSH or in a container**
`cv2.imshow` needs a display. Add `--no-show --save out.png`.

**Slow inference**
Check the selected device (section 2). CPU-only runs are 5 to 10× slower than MPS
or CUDA.
