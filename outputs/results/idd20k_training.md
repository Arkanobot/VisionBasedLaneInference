# Training on IDD Segmentation 20k

## What changed

The model reported elsewhere in this project is trained on **IDD-Lite**: 1,403
images at 320×227. This run uses **IDD Segmentation 20k** — 6,993 training and
981 validation images at full resolution, downscaled to 512×288 for training.
That is 5× the images at 2.3× the linear resolution.

Everything else is identical: the same architecture, the same augmentation, the
same distilled lane-line head, the same loss.

## Result

The schedule ran to completion, 40 epochs, best at epoch 35:

| | IDD-Lite | **IDD-20k** | change |
|---|---|---|---|
| mIoU | 0.6833 | **0.7574** | **+0.0741** (+10.8%) |
| drivable IoU | 0.9326 | **0.9538** | +0.0212 |
| lane-line IoU | 0.5480 | **0.6311** | **+0.0831** (+15.2%) |
| pixel accuracy | 0.8788 | **0.9148** | +0.0360 |

Training: 437 steps/epoch, batch 16 at 288×512, 127 minutes total on Apple MPS.
The run converged well before it ended — validation mIoU over the final five
epochs spans 0.0006.

The gains land where the earlier analysis predicted they would. The two classes
identified as data-starved on IDD-Lite were the rare ones, and the lane-line
head was described as resolution-limited because a 150 mm marking is sub-pixel
at 227 px tall. Both improved most: lane-line IoU by 15%, and overall mIoU by
more than ten percent relative, while drivable IoU — already at 0.93 and close
to its ceiling — moved comparatively little.

## Status of the weights

`seg_idd20k` is the **shipped checkpoint**, and every figure reported for the
system as delivered is measured against it.

Getting there took two attempts. The first, on Colab, was destroyed at epoch 26
of 40 when the free-tier GPU quota expired — the post-mortem below is kept
because the lesson generalises. The schedule was then rerun locally on Apple MPS
and completed all 40 epochs in 127 minutes, reaching a slightly better result
than the truncated run had (0.7574 against 0.7547).

Note that the 0.7574 mIoU here is measured on the **IDD-20k validation split**
(981 frames). Elsewhere in this project the same model measures 0.7133 mIoU on
the **IDD-Lite validation split** (204 frames), which is the split used for all
comparative evaluation. The two numbers are different splits, not a discrepancy.

## Post-mortem: why the weights were lost

Training wrote `last.pt` and `best.pt` to `/content/odp/outputs/checkpoints/`,
on Colab's local disk, and copied them to Drive only in a cell that ran *after*
training completed. Colab destroys `/content` when a session ends, and free-tier
GPU sessions end without warning when the quota expires. The session ended at
epoch 26 of 40, so the copy never ran.

`--resume` had been added specifically to make a dropped session cheap. It could
not help, because the checkpoint it resumes from was stored in the one location
a dropped session destroys. Adding resume while leaving the checkpoint on
ephemeral storage is not a partial safeguard; it is no safeguard at all.

Fixed by `--checkpoint-dir`, and the notebook now passes
`/content/drive/MyDrive/ODP/checkpoints`, so every epoch is durable as it
completes and `--resume` behaves as intended.

Two things limited the damage, both by luck rather than design:

* The prepared dataset had already been archived to Drive in an earlier cell, so
  the download, label rasterisation and downscaling — the slow, fragile part of
  the pipeline — did not need repeating. Only 62 minutes of GPU time was lost,
  not the ~2 hours the whole session took.
* The run had converged. Validation mIoU over the final five completed epochs
  was 0.7511, 0.7547, 0.7534, 0.7533, 0.7538 — a spread of 0.0036 — so the
  fourteen unrun epochs would have changed the outcome very little.

The complete 26-epoch Colab history was recovered from the session log and is
kept in `idd20k_history_colab.json`. `idd20k_history.json` now holds the 40-epoch
local run that produced the shipped weights.
