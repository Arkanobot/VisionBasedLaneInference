# Metric scale: analysis and calibration

## Why this needs care

Lane inference partitions the carriageway by the IRC design lane width, so it
consumes a *metric* road width. That width comes out of the inverse perspective
mapping, which in general needs the camera's focal length, its height above the
road, and its pitch. IDD-Lite ships no camera metadata, and its images have been
resampled to 320x227 from more than one source resolution, so nothing can be
read off the files.

## Finding 1: pitch does not have to be assumed

Pitch is *measured* per frame from the vertical position of the vanishing point:

    theta = atan((cy - v_vp) / f)

so it is an observation, not a prior.

## Finding 2: lateral scale is independent of focal length

This is the useful result. Substituting the measured pitch back into the
ground-plane projection and simplifying, the lateral extent on the ground
corresponding to a pixel width `du` observed at image row `v` is

    dX  =  du * H / (v - v_horizon)

where `H` is the camera height and `v_horizon` is the vanishing point's row.
**The focal length cancels.** It cancels because the horizon position already
encodes `f * tan(theta)` jointly, and that combination is exactly what sets the
ground-plane scale.

Verified numerically by holding the vanishing point fixed and sweeping the
assumed field of view over 40 deg to 140 deg:

| assumed FOV | focal (px) | derived pitch | lateral span recovered | forward distance |
|---|---|---|---|---|
| 40 deg  | 439.6 | 1.11 deg | 3.467 m | 7.60 m |
| 60 deg  | 277.1 | 1.76 deg | 3.468 m | 4.77 m |
| 80 deg  | 190.7 | 2.55 deg | 3.470 m | 3.25 m |
| 100 deg | 134.3 | 3.62 deg | 3.474 m | 2.25 m |
| 120 deg |  92.4 | 5.26 deg | 3.481 m | 1.50 m |
| 140 deg |  58.2 | 8.30 deg | 3.503 m | 0.84 m |

Lateral scale varies by **1% across a 100-degree range of assumed FOV**;
longitudinal distance varies by a factor of nine. Since lane inference is a
purely lateral operation, it is insensitive to the field of view. The dataset
measurement confirms this end to end: median vehicle width came out as 1.250 m,
1.252 m and 1.256 m at 70, 90 and 120 degrees respectively.

Consequence: the pipeline has exactly **one** metric free parameter, the camera
height. `fov_deg` is retained only because forward distance labels and the
extent of the rectified canvas depend on it; it is set to 90 deg.

## Finding 3: the remaining parameter cannot be pinned down from IDD alone

Two anchors were measured on the validation split using ground-truth masks.

**Carriageway width.** Median rectified width over 171 frames with at least six
unclipped samples: 4.06 m at H = 1.75 m. As first measured this statistic was
*biased low* for two separate reasons, and was therefore not used to set the
calibration:

  * a frame only contributes when both road edges are visible, which excludes
    the wide urban roads that fill the frame;
  * more importantly, the per-row longest contiguous run of road pixels is
    interrupted by any vehicle standing on the carriageway, so the width of a
    traffic gap was being recorded as the width of the road.

Correcting the second bias (see the addendum in `geometry_ablation.md`) raises
the median to **7.03 m** on raw model output, against the IRC:86 two-lane design
width of 7.0 m. The corrected anchor therefore *supports* H = 1.75 m rather than
implying the 2.76 m that the uncorrected figure suggested, and it now agrees
with the vehicle anchor instead of contradicting it.

**Vehicle ground-contact width.** For well-resolved vehicle components (area
>= 250 px, clear of the frame edge, plausible aspect ratio, ground contact in a
fixed image row band so the selection is itself FOV-free), the median rectified
width is 1.252 m at H = 1.75 m, from n = 16 instances.

Turning that into a height requires knowing what the median labelled vehicle
actually is, and IDD's level-1 `vehicles` class merges motorcycles (~0.7 m),
autorickshaws (~1.4 m), cars (~1.7 m), and buses (~2.5 m):

| assumed median vehicle | implied camera height |
|---|---|
| motorcycle-dominated, 1.10 m | 1.54 m |
| autorickshaw, 1.40 m | 1.96 m |
| passenger car, 1.70 m | 2.38 m |

With n = 16 these cannot be separated. A windscreen-mounted camera sits at
roughly 1.2-1.5 m and a roof or commercial-vehicle mount at 2.0-2.5 m.

## Adopted position

`camera_height_m = 1.75`, stated as a **prior, not a measurement**, sitting
between the autorickshaw and motorcycle anchors.

Because `dX` is exactly proportional to `H`, every metric quantity the pipeline
reports scales linearly with this number, and a reader can rescale any result by
inspection. Three things follow, and all three are implemented rather than
waved at:

1. Lane-count results are reported with a **sensitivity sweep** over
   H in {1.50, 1.75, 2.00, 2.40}, so it is visible which conclusions survive.
2. `IPMTransform.calibrate_from_known_width()` eliminates the parameter outright
   for any deployment where a single true width is known - one measured lane or
   carriageway fixes the scale exactly.
3. Scale-invariant results are reported wherever possible. Segmentation
   IoU/Dice, the width coefficient of variation used to score rectification
   quality, and the *ratio* of vehicle width to carriageway width (0.257) are
   all unaffected by this parameter entirely.

## What was wrong in an earlier draft of this analysis

An initial sweep appeared to show two independent anchors agreeing on a 104-deg
field of view. That sweep was invalid: it scaled a single FOV=60 measurement by
`tan(FOV/2)` instead of rebuilding the transform, which ignored the fact that
the derived pitch changes with `f` and very nearly cancels the effect. Rebuilding
the transform at each FOV showed the near-invariance documented above. The
apparent agreement was an artefact of the broken sweep, and the corrected
analysis replaces it.

## Addendum: the 7.03 m anchor is withdrawn

The 7.03 m median quoted above was produced by bridging vehicle occlusions
across every rectified row. That step is now known to inflate the measured
carriageway, because an occluder spanning a median joins the ego carriageway to
the opposing one (see the addendum in `lane_accuracy.md`). With the row-wise
merge the shipped pipeline gives a median of 5.23 m on the same split, so the
agreement with the IRC:86 7.0 m two-lane design width was an artefact of the
inflation and cannot be used to support H = 1.75 m.

It was also the wrong standard to reach for. Measured from the drivable ground
truth, only 22% of these validation frames carry a carriageway as wide as a
7.0 m two-lane section; most of this split is simply narrower road than that
standard describes.

What replaces it is a narrower claim. At H = 1.75 m the median inferred
carriageway is 5.23 m against 5.12 m measured from ground truth on the same
frames, a difference of 0.11 m. That validates the geometry and it does not
validate the height: the ground-truth width is recovered through the same
rectification and scales linearly with the assumed height exactly as the
inferred width does, so both move together under a different prior. The
absolute scale still rests on the assumption, and the sensitivity sweep in
`sensitivity.md` remains the honest treatment of it.
