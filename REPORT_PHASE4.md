# Study Project, Phase 4

## Vision-Based Lane Inference for Unstructured & Structured Indian Roads

**Course:** Bachelor of Science in Computer Science
**Students:** Shreyas Bhat K (2023EBCS460), Harshwardhan Mukund Mohadikar (2023EBCS353)
**Supervisor:** Dr. Ashok Yemineni

---

## 1. Introduction

### 1.1 Purpose of Phase 4

Phase 3 delivered a working prototype and an honest account of its weaknesses.
It could not, however, say how good it was. The evaluation was qualitative, seven images, inspected by eye, so there was no way to tell whether a change
was an improvement, and no basis on which to prioritise the remaining work.

Phase 4 has two objectives. The first is to replace qualitative assessment with
measurement: a labelled Indian-road dataset, a fixed evaluation protocol, and an
ablation in which the Phase 3 pipeline itself is one of the configurations
scored. The second is to close the gap that measurement exposes, and in
particular to deliver the *lane inference* the project is named for, which the
Phase 3 prototype approximated by bisecting the road.

### 1.2 Summary of work completed so far

**Phase 1** established scope, requirements and a modular architecture, from a
review of existing lane-detection work and its assumptions about road structure.

**Phase 2** built a classical computer-vision proof of concept, asphalt
segmentation, boundary extraction, lane approximation, and identified where
classical methods break down on Indian roads.

**Phase 3** added a pretrained deep-learning branch (YOLOP) alongside the
classical one, fused the two, and documented the resulting system's failure
modes: far-field instability, sensitivity to exposure and shadow, poor behaviour
in dense traffic, and inference latency of up to three seconds per frame.

**Phase 4**, reported here, rebuilds the system on a measured foundation while
keeping the architecture and module structure established in Phases 1 to 3.

---

## 2. implementation overview

### 2.1 What was built

| Module | Status | Function |
|---|---|---|
| `data/idd.py` | new | IDD-Lite loader; augmentation targeting Phase 3's documented failure modes |
| `models/segnet.py`, `train.py` | new | MobileNetV3-LR-ASPP-FPN segmentation network and training loop |
| `core/asphalt_detector.py` | rebuilt | adaptive CIELAB + texture road detection; Phase 3 version retained for ablation |
| `core/dl_drivable.py` | rebuilt | trained backend, corrected YOLOP reference backend, fusion strategies |
| `core/geometry.py` | new | vanishing-point estimation and metric inverse perspective mapping |
| `core/lane_tracker.py` | rebuilt | lane inference in the rectified view; Kalman smoothing; Phase 3 version retained |
| `core/semantics.py` | new | lane occupancy, traffic density, wrong-side detection |
| `core/pipeline.py` | new | stage orchestration with every stage independently switchable |
| `core/visualizer.py` | rebuilt | lane overlays, rectified panel, diagnostic dashboard |
| `eval/` | new | metrics, Phase 3 baselines, ablation harness, sensitivity sweep |
| `main.py` | extended | image / video / camera / dataset / benchmark; Phase 3 CLI still works |

### 2.2 Dataset

**IDD-Lite** (Indian Driving Dataset, level-1 label hierarchy): 1,403 training
and 204 validation frames of real Indian dashcam footage at 320×227, with
pixel-level ground truth over seven classes, drivable, non-drivable, living
things, vehicles, roadside objects, far objects, sky.

This addresses a genuine weakness of Phase 3, whose test set was a collection of
web images. Some carried stock-photo watermarks, and at least one was a European
mountain road, which is difficult to defend in a project about Indian roads.
Measured class distribution over the training split: drivable 32.4%, far objects
26.0%, sky 18.9%, roadside 11.2%, vehicles 8.1%, non-drivable 2.2%, living
things 1.3%.

### 2.3 The central change: lane inference

Phase 3 estimated lanes by taking the widest run of road pixels in each image
row and splitting it at the midpoint. This has two structural problems. It
returns exactly two lanes on every road, whatever its real width; and it
operates in image space, where the number of pixels per metre of road falls away
with distance.

The scale of that second problem is worth quantifying. A straight road of
constant 7 m width, viewed through the calibrated camera model, spans 385 pixels
at 5 m and 55 pixels at 35 m, a factor of seven. Any reasoning that treats
pixel counts as comparable across image rows is fighting that factor.

Phase 4 replaces it with a geometric pipeline:

1. **Vanishing point** from the intersection of the two road boundaries, robustly
   fitted. This requires no lane markings and therefore works on unstructured
   roads.
2. **Camera pitch** derived from the vanishing point's height, so pitch is
   measured per frame rather than assumed.
3. **Inverse perspective mapping** onto a metric ground plane. In this rectified
   view the same 7 m road is recovered as 7.0000 m at every distance.
4. **Lane inference.** Where painted markings are found they are used directly as
   lane boundaries. Where they are not, the unstructured case, the carriageway
   is partitioned into `round(width / nominal lane width)` equal lanes, with the
   nominal width taken from IRC:86 (3.5 m urban, 3.0 m rural). The integer
   rounding is what makes the output a lane *count* rather than an arbitrary
   subdivision.
5. **Ego lane** assignment from the camera's lateral position, then **Kalman
   smoothing** of the lane geometry and a majority vote on the discrete lane
   count across video frames.

---

## 3. System Validation and Testing

### 3.1 Protocol

All results are on the IDD-Lite validation split (204 frames), which was not
seen during training. Metrics are accumulated through a single dataset-wide
confusion matrix rather than averaged per image, which would inflate scores on
frames where a class is absent. Boundary F1 is reported at a 3-pixel tolerance
because IoU is insensitive to exactly the edge quality the downstream geometry
depends on.

Two deliberately trivial baselines are included to establish a floor. `all road`
predicts every pixel as drivable. `fixed trapezoid` uses a road-shaped region
with no image evidence at all. Any method that does not clearly beat these has
not learned anything about the specific scene in front of it.

### 3.2 Ablation, drivable-area detection

| Configuration | IoU | Precision | Recall | Boundary F1 |
|---|---|---|---|---|
| Trivial: everything is road | 0.3179 | 0.3179 | 1.0000 |, |
| Trivial: fixed trapezoid prior | 0.6450 | 0.8246 | 0.7476 | 0.1957 |
| Phase 3: HSV threshold | 0.4131 | 0.4682 | 0.7783 | 0.2093 |
| Phase 3: YOLOP as wired in Phase 3 | 0.1667 | 0.9119 | 0.1694 | 0.1382 |
| Phase 3: HSV OR YOLOP (the Phase 3 pipeline) | 0.4270 | 0.4747 | 0.8093 | 0.2048 |
| YOLOP, correctly wired | 0.7243 | 0.9923 | 0.7284 | 0.2205 |
| YOLOP correct + confidence blending | 0.6979 | 0.9181 | 0.7443 | 0.2324 |
| Phase 4: adaptive CIELAB + texture | 0.5697 | 0.7914 | 0.6840 | 0.2087 |
| **Phase 4: IDD-trained segmentation** | **0.9403** | **0.9700** | **0.9627** | **0.8343** |
| Phase 4: learned + cleanup | 0.9310 | 0.9760 | 0.9489 | 0.7754 |
| Phase 4: learned + confidence blending | 0.9066 | 0.9654 | 0.9410 | 0.7234 |
| Phase 4: learned + blending + cleanup | 0.9047 | 0.9721 | 0.9309 | 0.7031 |

Four findings follow from this table, and three of them were not expected.

**The Phase 3 classical detector loses to a fixed triangle.** At 0.4131 IoU the
HSV threshold performs worse than a hard-coded trapezoid that never looks at the
image (0.6450). Its precision of 0.4682 says that more than half the pixels it
called road were not road. The HSV bounds, all hues, saturation below 70,
value between 40 and 210, accept any desaturated, mid-brightness pixel, which
describes concrete, dust, walls, overcast sky and white vehicles as readily as
asphalt. The improved classical detector reaches 0.5697 and still does not clear
the trivial prior. This is the strongest available evidence for the project's
own thesis: classical colour-based road detection is not viable on Indian roads.

**YOLOP was not failing in Phase 3; it was mis-wired.** As Phase 3 used it,
YOLOP scores 0.1667 IoU at 0.1694 recall, it was finding about a sixth of the
road. Two preprocessing steps were missing: the input was stretched to 640×384
instead of letterboxed, and ImageNet normalisation was omitted entirely, so the
network received images from a distribution it had never been trained on.
Restoring both, with no retraining and identical weights, raises the same model
to **0.7243 IoU, a 4.3× improvement**. Phase 3's conclusion that "blind
implementation of DL would only introduce latency with only minor improvement"
was therefore an artefact of the integration, not a property of the approach.

**The bitwise-OR fusion could not have worked.** Phase 3 fused with
`cv2.bitwise_or` behind a `dl_ratio < 0.01` gate. A drivable-area head typically
claims 20 to 30% of the frame, so that gate almost never fires and the fusion was
in practice an unconditional OR, an operator that can only add pixels, and so
cannot suppress a false positive from either branch. The measurement bears this
out: HSV alone scores 0.4131 and HSV OR YOLOP scores 0.4270, so the deep-learning
branch contributed 0.0138 IoU to the Phase 3 system.

**Blending the two branches costs accuracy, in both regimes.** Replacing the OR
with a principled confidence-weighted sum still *reduces* IoU: 0.9403 → 0.9066
with the in-domain trained model, and 0.7243 → 0.6979 with out-of-domain YOLOP.
The reason is simply that the classical branch is much weaker than either
learned branch, so any fixed-weight blend drags the result toward it. The
hybrid CV+DL premise the project has carried since Phase 2 is therefore **not
supported by the data**. The classical branch is retained as a *gated fallback*, used only if the learned branch collapses, which is exactly equivalent to the
learned branch on this data while still covering model failure.

### 3.3 Semantic segmentation

Seven-class results, best checkpoint (epoch 88 of 150):

| Class | IoU | Dice | Precision | Recall |
|---|---|---|---|---|
| drivable | 0.9403 | 0.9663 | 0.9700 | 0.9627 |
| non-drivable | 0.4695 | 0.6390 | 0.5945 | 0.6907 |
| living things | 0.5268 | 0.6901 | 0.6445 | 0.7426 |
| vehicles | 0.7578 | 0.8622 | 0.8557 | 0.8687 |
| roadside objects | 0.4811 | 0.6496 | 0.6730 | 0.6278 |
| far objects | 0.7460 | 0.8545 | 0.8541 | 0.8548 |
| sky | 0.9435 | 0.9709 | 0.9648 | 0.9771 |
| **mean** | **0.6942** | **0.8047** | | |

Pixel accuracy 0.8820, mean class accuracy 0.8178. The weakest classes are the
two rarest, non-drivable at 2.2% of pixels and living things at 1.3%, which is
the expected consequence of class imbalance at this resolution.

### 3.4 Geometry

**Rectification is exact.** Round trips between image, ground and rectified
coordinates agree to 5×10⁻¹⁴ m, and a synthetic 7 m road is recovered as 7.0000 m
at every distance from 5 m to 35 m.

**Vanishing-point recovery: 186/204 frames (91.2%).** Getting there required one
non-obvious correction. Where the carriageway runs off the side of the frame,
the observed "road boundary" is the image border, which drives the edge fit
vertical and sends the vanishing point to infinity. On validation frame #101
this produced a 25.1° pitch estimate and a rectification containing only sky;
excluding border-clipped edge points corrects it to −0.9°.

**Rectification quality**, measured as the coefficient of variation of recovered
road width over 6 to 22 m, a scale-invariant statistic, so it is unaffected by the
metric calibration:

| Mask used for vanishing-point estimation | VP recovered | median width-CV | dense traffic |
|---|---|---|---|
| drivable (raw) | 184/204 | 0.2270 | 0.3094 |
| **drivable + filled holes (adopted)** | **186/204** | 0.2306 | 0.3039 |
| + bridge vehicle occlusions | 185/204 | 0.2414 | 0.3721 |
| road corridor (union with vehicles) | 152/204 | 0.3748 |, |

Two of these were hypotheses that the measurement rejected, and both are
reported rather than quietly dropped. Unioning vehicles into the road mask, reasoning that vehicles occlude the road rather than bound it, drops recovery
from 90.2% to 74.5%, because parked roadside vehicles extend the region sideways
into the frame border. Bridging only those gaps genuinely covered by occluder
pixels avoids that problem but still degrades width consistency in dense traffic,
because the bridged boundary is noisier to fit than the smaller unoccluded
fragment.

The same operation, applied instead to the **width-measurement** stage, is
essential. The per-row longest run of road pixels is interrupted by any vehicle
standing on the carriageway, so a pair of parked vans causes the gap between
them to be recorded as the width of the road, validation frame #0 reported a
2.75 m "carriageway" that was exactly that. Bridging occlusions for the width
raises lane-inference coverage from 150/204 to 171/204 and the median measured
carriageway from 5.24 m to 6.55 m. The two stages use different masks because
they ask different questions: where does the road converge, versus how wide is it.

### 3.5 Lane inference

| Quantity | Value |
|---|---|
| Frames with a lane solution | 171/204 (83.8%) |
| Ego lane assigned | 155/171 (90.6%) |
| Median carriageway width | 6.55 m |
| Median inferred lane width | 3.32 m |
| Mean lane count | 2.06 |
| Lane-count distribution | 1: 43, 2: 83, 3: 37, 4: 7, 5: 1 |
| Classified structured / unstructured | 16 / 155 |
| Mean confidence | 0.684 |

The median inferred lane width of 3.32 m sits between the IRC rural (3.0 m) and
urban (3.5 m) design widths. This is a meaningful consistency check rather than
a fitted result: lane width is an *output* of dividing a measured carriageway by
an inferred integer count, so its landing in the design range indicates that the
width measurement and the count are consistent with each other.

Only 16 of 171 frames were classified as structured. At 320×227 a 150 mm lane
marking is under half a pixel wide in the far field, so the marking detector is
operating at its resolution limit; this figure should improve substantially on
full-resolution input and is not evidence about Indian roads.

### 3.6 Performance

End-to-end, Apple M2 Pro (MPS), 60 frames after warm-up:

| Stage | Classical backend | Learned backend |
|---|---|---|
| segmentation |, | 13.15 ms |
| classical detection | 2.91 ms | 2.99 ms |
| fusion / gating | 0.03 ms | 0.09 ms |
| cleanup | 1.17 ms | 1.01 ms |
| geometry (VP + IPM) | 8.70 ms | 9.24 ms |
| lane inference | 2.74 ms | 5.17 ms |
| **total** | **15.55 ms (59.4 FPS)** | **31.65 ms (31.2 FPS)** |

Against Phase 3's reported 100 to 200 ms classical and up to 3 s with deep learning,
the complete Phase 4 pipeline, which does strictly more work, including
rectification and lane inference that Phase 3 did not perform, runs at 31 FPS.
Three changes account for it: a 3.32 M-parameter architecture chosen for
depthwise-separable efficiency, use of the available GPU rather than a hard-coded
CPU path, and removal of the `torch.set_num_threads(2)` cap. Training took 30.4
minutes on a Colab T4 (12.2 s/epoch, 150 epochs).

---

## 4. Metric Calibration and Its Limits

Lane inference consumes a metric road width, so scale matters. Two results
govern it.

**Lateral scale is independent of focal length.** With pitch derived from the
observed vanishing point, the lateral ground extent of a pixel width `du` at
image row `v` reduces to `du · H / (v − v_horizon)`, where `H` is camera height.
The focal length cancels, because the horizon position already encodes
`f·tan(pitch)`. Verified by sweeping assumed field of view from 40° to 140° with
the vanishing point held fixed: recovered lateral span moves from 3.467 m to
3.503 m, 1%, while forward distance changes by a factor of nine. Since lane
inference is purely lateral, it is insensitive to field of view, and calibration
reduces to a single parameter.

**That parameter cannot be determined from IDD-Lite alone.** The dataset carries
no camera metadata. Two anchors were measured: the median carriageway width
(7.03 m on raw model output, against the IRC two-lane design width of 7.0 m) and
the median well-resolved vehicle ground-contact width (1.25 m, from n=16
instances, a plausible median for traffic in which motorcycles and
autorickshaws are numerous). Both are consistent with the adopted camera height
of **1.75 m**, but neither pins it down: IDD's level-1 `vehicles` class merges
motorcycles, autorickshaws, cars and buses, so the vehicle anchor alone spans
1.54 to 2.38 m depending on what the median vehicle is assumed to be.

The camera height is therefore stated as a **prior, not a measurement**. Every
metric output scales linearly with it. Sensitivity across the plausible range:

| Camera height | Frames solved | Median carriageway | Median lane width | Mean lane count |
|---|---|---|---|---|
| 1.30 m | 153/204 | 5.64 m | 3.29 m | 1.87 |
| 1.50 m | 167/204 | 6.08 m | 3.16 m | 1.98 |
| **1.75 m (adopted)** | **171/204** | **6.54 m** | **3.26 m** | **2.07** |
| 2.00 m | 177/204 | 7.22 m | 3.45 m | 2.15 |
| 2.40 m | 172/204 | 7.93 m | 3.39 m | 2.32 |

Median lane width stays within 3.16 to 3.45 m across the entire sweep. This is a
real robustness property of integer partitioning: as the assumed scale grows,
the inferred lane count grows with it and the quotient stays near the design
width. The *structure* the system reports degrades gracefully under scale error
even though absolute widths do not. `IPMTransform.calibrate_from_known_width()`
eliminates the parameter entirely for any deployment where one true width is
known.

---

## 5. risk review

| Risk (from Phase 3) | Status |
|---|---|
| Team unfamiliar with deep learning | **Closed.** A network was designed, trained and evaluated end to end on domain data. |
| Dependence on third-party frameworks | **Accepted, reduced.** PyTorch remains a dependency; YOLOP is now a baseline rather than a component. |
| Classical CV insufficient | **Confirmed and quantified.** 0.5697 IoU, below a trivial trapezoid prior at 0.6450. |
| Dataset availability | **Closed.** IDD-Lite provides 1,607 labelled Indian road frames. |
| Complex road geometry | **Partly addressed.** Lane counts from 1 to 5 are inferred; junctions remain unhandled. |
| Time and resource constraints | **Managed.** Training is 30 min on free Colab; inference runs on a laptop. |

---

## 6. Limitations

- **Metric scale rests on an assumed 1.75 m camera height** (Section 4). All
  metric outputs scale linearly with it.
- **16% of validation frames yield no lane solution**, most often where the carriageway is
  too wide to fit within the frame, so neither observed boundary is a road
  edge. Traffic density does not explain failure (8.8% vehicle pixels in
  unsolved frames against 8.5% in solved). Single-frame
  geometry cannot recover information that is not in the frame; the temporal
  filter addresses this for video but was not evaluated on video, as IDD-Lite
  contains independent frames rather than sequences.
- **The ground plane is assumed flat.** Crests, dips and banked curves violate the
  homography.
- **Structured-road classification is resolution-limited.** Only 16/171 frames were
  classified structured, reflecting 320×227 input rather than Indian road marking
  practice.
- **Wrong-side detection is implemented but unvalidated.** It needs motion, so it
  is video-only, and IDD carries no ground truth for it. It is reported as an
  experimental capability with its failure modes documented in code.
- **Rare classes are weak** (non-drivable 0.4695, roadside objects 0.4811).
- **No test-split evaluation.** IDD-Lite's 404 test frames are unlabelled, so all
  results are on the validation split. It was not used for model selection
  beyond choosing the best epoch by mIoU, which is a mild optimistic bias.

---

## 7. future work

- **Video validation.** The Kalman filter and lane-count vote are implemented and
  exercised but not quantitatively evaluated, for want of labelled Indian video.
  Recording and annotating even a short sequence would let temporal stability be
  measured rather than assumed.
- **Full-resolution IDD.** IDD-Segmentation at native resolution should
  substantially improve rare-class IoU and lane-marking detection.
- **Junction handling.** The single-carriageway model has no representation for
  intersections, where the road opens in several directions at once.
- **Learned lane boundaries.** Adding a lane-marking segmentation head would let
  structured roads be handled by the model rather than by a hand-written filter.
- **Removing the height prior**, either by calibrating against a known width at
  deployment or by estimating height from tracked vehicle dimensions over time.

---

## 8. learning outcomes

The most useful lesson of this phase was methodological. Phase 3 concluded that
deep learning offered little benefit for the latency it cost. That conclusion was
reasonable given the evidence available, and it was wrong, the model had been
fed stretched, un-normalised input, and correcting the preprocessing alone
improved it 4.3×. The error was invisible because there was no measurement that
could have exposed it. Building the evaluation harness before attempting
improvements changed how every subsequent decision was made.

The second lesson concerns negative results. Three plausible ideas were
implemented and then rejected by measurement: unioning vehicles into the road
mask, bridging occlusions for vanishing-point estimation, and blending the
classical and learned branches. Each is retained in the codebase with its
numbers. The occlusion idea then turned out to be correct for a *different*
stage, which would not have been discovered had it been deleted after the first
negative result.

Technically, the phase covered training and evaluating a semantic segmentation
network on domain data, projective geometry and camera calibration from a single
view, recursive state estimation, and the discipline of building a system whose
stages can be independently switched off so that each one's contribution can be
attributed.

---

## 9. Deliverables

- **Source code**, modular pipeline; `README.md` for design, `guide.md` for setup
  and run instructions.
- **Trained model**, `outputs/checkpoints/seg_mnv3/best.pt`, 3.32 M parameters,
  0.6942 mIoU on IDD-Lite validation.
- **Colab training notebook**, `dist/train_colab.ipynb`, reproduces the model on a
  free T4 in ~30 minutes.
- **Evaluation harness**, `python -m eval.evaluate` regenerates every table in
  this report.
- **Measured results and analyses**, `outputs/results/`.
- **Demo**, `python main.py video <file> --panel`.

---

## 10. Conclusion

Phase 4 set out to measure the system and then to improve it. Both were done, and
the measurement mattered more than expected.

On drivable-area detection the system moves from **0.4270 IoU** for the Phase 3
pipeline to **0.9403**, a 2.2× improvement, while the complete pipeline, which
also performs rectification and lane inference that Phase 3 did not
attempt, runs at 31 FPS against Phase 3's worst case of one frame every three
seconds. Lane inference produces a metric lane structure on 83.8% of validation
frames, with a median inferred lane width of 3.32 m against the IRC design range
of 3.0 to 3.5 m.

Two of the project's standing assumptions did not survive contact with the data.
Deep learning was not marginally useful, as Phase 3 concluded; it had been wired
incorrectly, and correcting only the preprocessing improved the same pretrained
weights by 4.3×. And the CV+DL hybrid the project has pursued since Phase 2 does
not help: with a properly trained in-domain model, blending in the classical
branch measurably reduces accuracy, and the classical component's real value is
as a fallback rather than as a fusion partner.

The remaining limitations are stated in Section 6 and are, for the most part,
limitations of the data rather than the method. The largest, that metric scale
rests on an assumed camera height, has a known remedy that requires one
measured width at deployment.

---

## 11. Supervisor Review and Approval

Advisor Feedback:

Supervisor Comments:

Recommendations:

Signature: ___________________________

Date: _______________________________
