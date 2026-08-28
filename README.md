# Vision-Based Lane Inference for Unstructured & Structured Indian Roads

**BITS Pilani, BSc Computer Science, Study Project, Phase 4**
Shreyas Bhat K (2023EBCS460) · Harshwardhan Mukund Mohadikar (2023EBCS353)
Supervisor: Dr. Ashok Yemineni

---

## What this does

Infers lane structure from a single forward-facing road image, on roads that
have no lane markings at all.

The distinction between *lane detection* and *lane inference* is the point. On a
marked highway you can detect the paint. On an unmarked Indian road there is no
paint, yet the road still carries a lane structure that drivers agree on. This
system infers that structure from the geometry of the drivable surface plus a
prior on how wide a lane is:

1. segment the scene into 7 semantic classes with a network trained on the
   **Indian Driving Dataset**;
2. estimate the **vanishing point** from the road boundaries, which needs no
   markings;
3. rectify the road to a **metric bird's-eye view**, where a metre is a metre
   regardless of distance;
4. measure the carriageway width and partition it into
   `round(width / IRC design lane width)` lanes;
5. assign the ego lane, then smooth the whole model across video frames with a
   Kalman filter.

## Headline results

Measured on the IDD-Lite validation split (204 frames). Every number here is
produced by the evaluation harness in `lane_inference/eval/` and can be
re-run; nothing is typed in by hand.

Drivable-area segmentation, against the approaches this replaces:

| Configuration | Drivable IoU | Boundary F1 |
|---|---|---|
| Trivial: fixed trapezoid, ignores the image | 0.6450 | 0.1957 |
| v1/v2 classical (HSV threshold) | 0.4131 | 0.2093 |
| v2 classical (adaptive CIELAB + texture) | 0.5697 | 0.2087 |
| v3 YOLOP, as first wired | 0.1667 | 0.1382 |
| v4 YOLOP, correctly wired | 0.7243 | 0.2205 |
| **v6 IDD-trained segmentation (this project)** | **0.9403** | **0.8343** |
| **v6 as deployed, with mask cleanup** | **0.9310** | **0.7754** |

Seven-class semantic segmentation: **mIoU 0.7133**, pixel accuracy 0.8920.

Lane inference, against drivable-class ground truth over 2,288 boundary
samples on 159 comparable frames:

| Quantity | median | mean | p90 |
|---|---|---|---|
| Road-edge error | 0.314 m | 0.639 m | 1.526 m |
| Centreline error | 0.292 m | 0.544 m | 1.163 m |
| Carriageway width error, relative | 11.8% | 22.4% | 46.5% |

A lane solution is produced on **165 of 204** frames and refused on 39. The
refusals are deliberate: a wrong lane model is worse than none, and the
system declines when the carriageway falls below a 2.5 m plausibility floor,
when the road edges are not visible, or when no vanishing point is recovered.

Full pipeline latency on an Apple M2 Pro (MPS), including geometry and lane
inference: **36.5 ms, about 27 fps**. Segmentation alone is 15.8 ms on the
GPU and 75 ms on two CPU threads.

Temporal filtering over 150 clips cuts width jitter from 0.311 m to 0.038 m
and lane-count flips from 20.1 to 4.3 per 100 frames.

## How this was arrived at

The project ran as a numbered progression of versions, each started because
the one before it had been measured and found short:

| Version | Approach | Why it was superseded |
|---|---|---|
| `lane_detector_project` | Classical colour and texture segmentation | The drivable surface is not separable from shadow, wet tarmac and unpaved shoulder by colour alone |
| `v2` | HSV thresholding plus asphalt heuristics | Each heuristic fixed one scene and broke another |
| `v3` | YOLOP dropped in as the detector | Scored below the fixed-trapezoid baseline |
| `v4` | YOLOP with letterboxing and normalisation corrected | Large gain, confirmed the wiring, still short of what the geometry needs |
| `v5_dl` | First segmentation network trained on IDD | Trained on a partial dataset root |
| `v6_dl_fullDataset` | Shipped architecture on the full IDD-20k split | Current |

Every one of those approaches is carried into the ablation above and measured
on the same split with the same metrics, rather than described and discarded.

## Quick start on MacOS

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r lane_inference/requirements.txt

cd lane_inference
python main.py image ../datasets/idd_lite/leftImg8bit/val/119/903127_image.jpg --panel
```

Everything below assumes you are in `lane_inference/` with the venv active.

## Usage

```bash
# single image, with the diagnostic panel
python main.py image road.jpg --panel

# video, with temporal smoothing (enabled automatically)
python main.py video drive.mp4 --save annotated.mp4 --panel

# live camera
python main.py camera 0 --panel

# a grid of dataset frames
python main.py dataset --split val --count 8 --save grid.png

# per-stage latency breakdown
python main.py benchmark --backend all
```

Useful flags: `--backend {cv,dl,hybrid}`, `--camera-height` (the one metric
calibration parameter), `--max-range`, `--no-show` for headless runs.

## Reproducing the results

```bash
# 1. data: register at https://idd.insaan.iiit.ac.in/, accept the licence,
#    download IDD-Lite, then verify your copy against our manifest
python tools/fetch_dataset.py --archive <downloaded archive>

export PYTHONPATH=lane_inference

# 2. every measured result
python -m eval.evaluate           # segmentation and the twelve-way ablation
python -m eval.lane_geometry      # carriageway accuracy against ground truth
python -m eval.summary            # descriptive lane statistics
python -m eval.temporal           # stability over 150 clips
python -m eval.sensitivity        # camera-height sweep
python -m eval.failure_analysis   # the refusal breakdown

# 3. the test suite
python -m pytest lane_inference/tests -q      # 55 tests
```

Each evaluation rewrites the matching file under `outputs/results/`, and the
application reads those files at run time, so re-running a measurement updates
the interface to match.

The occlusion-handling ablation is reproduced with `PROFILE_MODE=raw` or
`PROFILE_MODE=bridge` before the `eval.lane_geometry` command.

Training takes about 30 minutes on a Colab T4. `dist/train_colab.ipynb` runs it
end to end and writes the checkpoint back to Drive.

## Deliverables

| File | What it is |
|---|---|
| `outputs/Capstone_Report.pdf` | Full report, 61 pages |
| `outputs/Capstone_Report_Concise.pdf` | Concise edition, 32 pages |
| `outputs/Capstone_Presentation.pptx` | 24-slide deck |
| `handover/VIVA_DEFENCE.md` | Defence notes: headline numbers, hard questions, limitations |
| `outputs/results/` | Every measured result, read live by the application |

The dataset is not redistributed here. IDD is released through registration
with its licence presented at download time; `tools/fetch_dataset.py` verifies
a reader's own copy against the checksums every reported result was measured
on.

## Layout

```
lane_inference/
  config.py              configuration; Phase 3 constants preserved for the ablation
  main.py                CLI: image / video / camera / dataset / benchmark
  core/
    asphalt_detector.py  classical road detection (Phase 4 adaptive + Phase 3 legacy)
    dl_drivable.py       trained backend, YOLOP reference backend, fusion
    geometry.py          vanishing point, inverse perspective mapping
    lane_tracker.py      lane inference in the rectified view, Kalman smoothing
    semantics.py         lane occupancy, traffic density, wrong-side detection
    pipeline.py          stage orchestration; every stage independently switchable
    visualizer.py        overlays, rectified panel, diagnostic dashboard
  models/                network architecture and training loop
  data/idd.py            IDD-Lite loader and augmentation
  eval/                  metrics, ablation baselines, evaluation, sensitivity
outputs/results/         measured results and the analyses behind them
```

## What changed since Phase 3, and why

Phase 3 concluded that deep learning added latency without much accuracy. That
conclusion was an artefact of how the model was wired, not a property of the
approach. Correcting the preprocessing alone, letterboxing and ImageNet
normalisation, both of which were missing, takes the same pretrained YOLOP
weights from **0.1667 to 0.7243 IoU**, a 4.3× improvement with no retraining.

Three further findings are documented with their measurements in
`outputs/results/`:

- **Lateral metric scale is independent of focal length.** When pitch is derived
  from the observed vanishing point, `f` cancels; the recovered lateral scale
  varies 1% across a 100° sweep of assumed field of view. Calibration reduces to
  one parameter, the camera height.
- **Confidence blending of the classical and learned branches costs accuracy**,
  in-domain (0.9403 → 0.9066) and out-of-domain (0.7243 → 0.6979) alike. The
  classical branch is simply weaker, so any fixed-weight blend drags the result
  down. It is kept as a *gated fallback* instead, which is exactly equivalent to
  the learned branch on this data while still covering model collapse.
- **Vehicle occlusion has to be handled per rectified row, not per frame.**
  Bridging a vehicle-shaped gap in the road mask is right when the gap is a
  vehicle on the carriageway and wrong when the occluder spans a median, where
  it joins the ego carriageway to the opposing one and the measured road
  doubles. Bridging every row took the relative width error at p90 from 34% to
  151%. Choosing per row brought it to 47% while keeping most of the coverage.
  The defect was found by comparing the lane model against the mask it was
  built from rather than against ground truth: the mask was within 0.20 m of
  truth, the model built from it was 0.58 m wider.

## Limitations

- **Metric scale rests on an assumed camera height of 1.75 m.** All metric
  outputs scale linearly with it. `IPMTransform.calibrate_from_known_width()`
  removes the assumption given one known width. See
  `outputs/results/scale_calibration.md`, including the sensitivity sweep.
- **Lane inference refuses 39 of 204 validation frames**, by design: 20 where
  the carriageway measures below the 2.5 m floor, 11 where the road edges are
  not visible, 8 where no vanishing point is recovered. Traffic density
  separates frames that get an answer from those that do not (10.2% vehicle
  pixels against 8.2%) but no longer separates accurate answers from
  inaccurate ones.
- **Lane counts are inferred, not detected, and have no ground truth.** The
  carriageway extent is validated; its subdivision on unmarked roads cannot be,
  because no Indian dataset carries lane annotations.
- **Ego lane and road-user distances are reported but never evaluated.** IDD
  has no ground truth for either; read them as indicative.
- **The ground plane is assumed flat.** Crests, dips and banked curves violate it.
- **Wrong-side detection is experimental and unvalidated.** IDD carries no
  ground truth for it; its failure modes are documented on the class.
- **IDD-Lite is 320×227.** The lane-marking detector is near its resolution
  limit, so structured-road classification is weaker than it would be on
  full-resolution input.

## Acknowledgements

Indian Driving Dataset (IDD), IIIT Hyderabad, obtained from
<https://idd.insaan.iiit.ac.in/>. Used under the licence presented at download;
not redistributed here.

YOLOP (Wu et al., 2022) is used offline only, as a baseline and as the teacher
for the distilled lane-marking head. Its weights are not redistributed.

MobileNetV3 (Howard et al., 2019) weights come from torchvision.

Supervised by Dr. Ashok Yemineni, BITS Pilani.
