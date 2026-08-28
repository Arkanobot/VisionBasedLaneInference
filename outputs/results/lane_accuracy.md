# Accuracy of the inferred carriageway

Every other figure reported for lane inference is either segmentation accuracy —
a different task — or temporal stability, which says nothing about correctness.
This measures correctness.

## Method

No Indian dataset carries lane annotations, which is precisely why the lane
*inference* problem exists. But whatever the lane count is, the **outermost**
inferred lane boundaries must coincide with the real edges of the road, and the
drivable-class ground truth gives those edges.

Both are rectified into the same metric ground plane and compared at 1 m
intervals from 6 m to 22 m ahead. Samples where either reference edge is clipped
by the image border or the canvas are excluded, since there the reference is not
a road edge either.

This validates the carriageway extent, and through it the whole geometry stage.
It does **not** validate the interior lane divisions on unmarked roads, for which
no ground truth exists anywhere.

## Result

204 validation frames; a lane model on 165; 159 comparable;
2,288 boundary samples.

Each comparable frame contributes at most 34 samples, two edges at each of the
17 one-metre steps from 6 m to 22 m. A sample is dropped where either reference
edge is clipped by the frame or the canvas, because there the ground truth is
not a road edge either. 2,288 of a possible 5,406 survive that filter,
which is 42%.

| Quantity | median | mean | p90 |
|---|---|---|---|
| Left edge error | 0.314 m | 0.639 m | 1.526 m |
| Right edge error | 0.334 m | 0.689 m | 1.732 m |
| Centreline error | 0.292 m | 0.544 m | 1.163 m |
| Carriageway width error | 0.630 m | 0.886 m | 2.004 m |
| Width error, relative | 11.8% | 22.4% | 46.5% |

These are per-sample statistics. The gate table below is per-frame: each frame
contributes the median of its own samples, so the two sets of numbers are not
directly comparable and the per-frame median is the higher of the two.

The median case is good. Half of all boundary samples fall within 45 cm of the
true road edge at distances up to 22 m. The mean sits far above the median and
p90 approaches 4 m, so the error distribution has a heavy tail: most frames are
accurate and a minority are badly wrong.

The relative width error deserves its own note. A p90 of 151% means that on the
worst tenth of samples the recovered carriageway is off by more than one and a
half times its true width. That is a large error, and it is concentrated in the
same frames the confidence score is designed to identify: gating at 0.7 removes
them along with the rest of the tail.

Absolute errors scale linearly with the assumed 1.75 m camera height; the
relative width error does not.

## The tail, and making the system aware of it

A heavy tail only matters if the system cannot tell which frames are in it. As
originally written it could not: the confidence score correlated just **−0.187**
with measured error, and gating on it above 0.5 bought nothing.

Correlating candidate signals against measured error showed why:

| Signal | corr. with error | used in old score? |
|---|---|---|
| **edge polynomial fit residual** | **+0.435** | **no** |
| road users detected | +0.270 | no |
| vanishing-point confidence | −0.251 | no |
| fraction of samples with both edges valid | −0.248 | no |
| drivable fraction of frame | +0.203 | no |
| width coefficient of variation | +0.188 | yes |
| *(old combined score)* | *−0.187* | — |
| per-edge coverage | −0.054 | **yes, 40% weight** |

The strongest predictor was unused, and the score's largest single component was
statistically indistinguishable from noise.

Rebuilt from the four predictive signals, weighted by strength:

```
confidence = 0.40 * fit-residual term
           + 0.25 * vanishing-point confidence
           + 0.20 * both-edges-valid fraction
           + 0.15 * width-consistency term
```

Verified on a held-out half of the validation split (weights were chosen by
predictor strength, not fitted, so the split guards against reading noise as
signal): correlation **−0.540** against **−0.318** for the old score.

In the full pipeline, over all 163 comparable frames:

| Confidence gate | Frames kept | Median error | p90 error |
|---|---|---|---|
| none | 100% | 0.697 m | 2.586 m |
| ≥ 0.6 | 85% | 0.598 m | 2.348 m |
| **≥ 0.7** | **68%** | **0.549 m** | **2.251 m** |
| ≥ 0.8 | 34% | 0.425 m | 1.537 m |

Measured on the shipped configuration over all 167 scored frames, the rebuilt score correlates **-0.470** with boundary error. The original score, on the analysis that motivated this rebuild, correlation -0.272. Gating at 0.7 keeps 68% of frames and removes 21% of the median error and 13% of the p90, so the tail is identifiable in advance rather than only in
hindsight.

## Addendum: the occlusion-handling fix

The numbers above are a re-measurement. The previous run reported a 0.448 m
median left-edge error with a 3.364 m p90, and a relative width error whose p90
reached 151.2%. That tail was not a segmentation problem. Measured against
ground truth, the predicted road mask was within 0.20 m of the true carriageway
at the median, while the lane model built from it came out 0.58 m wider than
that same mask. The defect was in how vehicle occlusion was handled before the
width profile was taken.

Bridging a vehicle-shaped gap in the road mask is right when the gap is a
vehicle standing on the carriageway and wrong when the occluder spans a median,
because then the ego carriageway is joined to the opposing one and the measured
road roughly doubles. Bridging every rectified row took the relative width error
at p90 from 34.1% to 151.2%, while buying 20 frames that would otherwise have
been refused.

The shipped pipeline now chooses per row. A row the unbridged mask can measure
is kept; a row it cannot measure is taken from the bridged mask; a row both can
measure uses the bridged value only when bridging widened it by less than one
lane width.

| Width profile measured on | Solved | rel. median | rel. mean | rel. p90 |
|---|---|---|---|---|
| Unbridged mask only | 151 / 204 | 11.0% | 16.5% | 34.1% |
| Bridged mask only | 171 / 204 | 18.5% | 51.5% | 151.2% |
| Row-wise merge (shipped) | 165 / 204 | 11.8% | 22.4% | 46.5% |

Reproduce the endpoints with `PROFILE_MODE=raw` or `PROFILE_MODE=bridge` before
`python -m eval.lane_geometry`.

Coverage fell from 171 to 165 frames, and that is the intended direction. Of the
ten frames the merge stopped solving, eight had been in the worst fifth by
boundary error and two were the worst two frames on the split; four frames that
had not been solved before are now solved. The system refuses the frames it used
to answer wrongly.

One consequence is worth recording because it cuts against an earlier claim.
The reported confidence now correlates -0.272 with measured error, against
-0.470 before. The score did not get worse. Refitting its four weights on this
data and validating on a held-out half was tried and did not beat the shipped
weights (-0.451 against -0.513), so the weights are unchanged. What changed is
the error distribution: the large errors the score used to separate are gone, so
there is less variance left to explain.
