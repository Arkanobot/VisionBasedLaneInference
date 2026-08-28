# Multi-task training: distilling a lane-line head

## What was done

IDD carries no lane-marking annotation. YOLOP ships a trained lane-line
segmentation head, which the Phase 3 pipeline never read — it used only the
drivable-area head. Measured on the IDD-Lite validation split, YOLOP's lane head
returns lane-line pixels on 67% of frames against 9.4% for the hand-written
top-hat filter that Phase 4 initially used.

YOLOP is, however, ~150 ms/frame on CPU against 13 ms for our network, and is
trained on BDD100K rather than Indian roads. So its lane-line output was used as
a teacher: 1,607 pseudo-labels were generated over IDD-Lite train and val (77.2%
contain lane pixels, at 0.85% pixel share), and a second head was added to our
network and trained against them jointly with the semantic task.

## Cost and benefit

| | single-task | multi-task |
|---|---|---|
| mIoU | **0.6943** | 0.6833 |
| drivable IoU | **0.9355** | 0.9326 |
| pixel accuracy | **0.8822** | 0.8788 |
| lane-line IoU | — | **0.5480** |
| best epoch | 88 / 150 | 109 / 150 |

Per-class IoU moves by −0.0015 to −0.0050 on five of the seven classes. The
exception is `non-drivable` at −0.0497, which is also the rarest class after
`living-things` and the least stable between runs.

**Important caveat.** The two runs do not differ only in the objective. The
single-task model was trained on a Colab T4 at batch 32 and lr 6e-4; the
multi-task model on Apple MPS at batch 16 and lr 4e-4, because the lane labels
were generated later and locally. The 0.011 mIoU gap is therefore not cleanly
attributable to multi-task interference — some or all of it may be the
hyperparameters. A controlled comparison would retrain both under identical
settings, which has not been done.

## What the lane head actually buys

The point of the head is structured-road detection, and measuring that exposed
two defects in the code consuming it.

| Configuration | Structured frames | Source breakdown |
|---|---|---|
| top-hat filter, every peak treated as a divider | 16 / 171 (9.4%) | top-hat 16 |
| + edge/divider role classification | 27 / 171 (15.8%) | top-hat 27 |
| **+ learned head with detector fallback** | **39 / 173 (22.5%)** | learned 16, top-hat 23 |

**Defect 1 — every detection was treated as a lane divider.** A lane-line head
detects the road's outer boundaries as well as its interior dividers; YOLOP does
this, and the distilled head inherits it. An edge detection landing a few tens of
centimetres inside the segmentation boundary produced "lanes" 0.53–0.93 m wide,
and the plausibility gate then discarded the entire frame. Of 122 frames with
detected peaks, only 47 had any peak surviving the inside-carriageway test, and
only 5 passed the gap gate.

Fixed by separating detections by role: those within half a lane width of an
outer boundary are edge detections and are dropped, since the boundary is already
known more precisely from the drivable mask; detections closer together than half
a lane are merged, being one marking seen twice rather than two lanes.

**Defect 2 — a failed learned detection blocked the classical one.** The learned
head was tried first and, if it produced any peaks, the top-hat filter was never
attempted — even when the learned peaks failed validation. That cost five
structured frames the filter alone would have found. The detectors are now a
fallback chain, first plausible result wins.

The two detectors turn out to be **complementary**: the learned head succeeds on
16 frames the filter misses, the filter on 23 the learned head misses. That is
the argument for keeping both rather than replacing one with the other.

## Adopted

The multi-task checkpoint is the default (`config.CHECKPOINT_PREFERENCE`). It
trades 0.011 mIoU and 0.003 drivable IoU for a 44% increase in structured-road
detection (27 → 39 frames). For a project whose subject is lane inference rather
than semantic segmentation, that is the right side of the trade.

Full-pipeline figures with the multi-task model, IDD-Lite val:

| Configuration | Drivable IoU | Precision | Recall | mIoU | Boundary F1 | FPS |
|---|---|---|---|---|---|---|
| learned segmentation | 0.9310 | 0.9716 | 0.9571 | 0.6826 | 0.8064 | 39.6 |
| + cleanup | 0.9227 | 0.9786 | 0.9416 | — | 0.7589 | 72.8 |
| + CV blending | 0.9050 | 0.9678 | 0.9331 | — | 0.7167 | 63.5 |
| + CV blending + cleanup | 0.9010 | 0.9754 | 0.9219 | — | 0.6995 | 58.7 |

The ordering matches the single-task result: every post-processing stage costs
accuracy, and blending the classical branch costs the most.
