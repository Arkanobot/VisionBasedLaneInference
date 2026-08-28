# Vanishing-point / rectification ablation

Measured on the IDD-Lite **validation split** (204 frames), using ground-truth
segmentation masks so that the geometry stage is evaluated independently of
model error.

**width-CV** = std/mean of the recovered carriageway width sampled every 1 m
over 6–22 m in the rectified view. A correct rectification of a straight road
gives a near-constant width, so lower is better. The statistic is a ratio and is
therefore invariant to the metric-scale calibration.

| Mask used for VP estimation | VP recovered | median width-CV | width-CV, dense traffic |
|---|---|---|---|
| A. drivable (raw)                  | 184/204 (90.2%) | 0.2270 | 0.3094 (n=49) |
| **B. drivable + fill_holes**       | **186/204 (91.2%)** | 0.2306 | 0.3039 (n=48) |
| C. + bridge occlusions (vehicles)  | 185/204 (90.7%) | 0.2414 | 0.3721 (n=38) |
| D. + bridge occlusions (veh+people)| 185/204 (90.7%) | 0.2526 | 0.4447 (n=40) |
| E. road corridor (union vehicles)  | 152/204 (74.5%) | 0.3748 | — |

Dense traffic = frames where vehicles occupy >12% of pixels (60/204 frames).

## Findings

1. **Border-clipped edge points must be excluded.** Where the carriageway runs
   off the side of the frame, the observed boundary is the image border rather
   than the road boundary, which drives the edge fit vertical. On validation
   frame #101 this produced a 25.1° pitch estimate and a rectification
   containing only sky; excluding clipped points corrects it to −0.9°.

2. **Unioning vehicles into the road mask is actively harmful** (variant E):
   VP recovery falls from 90.2% to 74.5%. Parked roadside vehicles extend the
   region sideways into the frame border, creating exactly the clipping that
   finding 1 removes. The initial hypothesis — that vehicles fragment the mask
   and should be filled back in — was wrong in this form.

3. **Semantically-targeted occlusion bridging also fails** (variants C, D).
   Bridging only those gaps genuinely covered by occluder pixels avoids the
   border problem, but still degrades width-CV in dense traffic (0.309 → 0.372).
   The bridged boundary follows the outer, partly-occluded road edge, which is
   noisier to fit than the smaller but cleaner unoccluded fragment. Retained in
   the codebase, disabled by default.

4. **Dense traffic is not solvable from a single frame.** When the carriageway
   is not visible, its edges cannot be estimated, and no masking strategy
   recovers that information. The correct treatment is temporal: hold the
   previous estimate and gate on confidence (`VanishingPointTracker`), which is
   evaluated separately on video.

**Adopted configuration: variant B** — drivable mask with enclosed holes filled,
border-clipped edge points excluded, pitch bounded to ±15°.

---

## Addendum: occlusion bridging helps width, having failed for the vanishing point

Findings 3 above rejected occlusion bridging. That rejection was measured on the
*vanishing-point* stage and it stands. Re-testing the same operation on the
*width-measurement* stage, once the trained segmentation model made reliable
vehicle masks available, gave the opposite result.

The cause of the original narrow-width readings turned out not to be the metric
calibration at all. `extract_road_profile` takes the longest contiguous run of
road pixels in each rectified row, so a pair of parked vans reduces the measured
carriageway to the width of the gap between them. Validation frame #0 reported a
2.75 m "carriageway" that was in fact the space between two parked vans.

Measured on the 204-frame validation split, using the trained model, road-only
mask for the vanishing point in every row:

| carriageway measured on | lane solutions | median carriageway |
|---|---|---|
| road mask only | 150/204 (73.5%) | 5.24 m |
| **road mask, occlusions bridged** | **171/204 (83.8%)** | **6.55 m** |

Isolated on raw model output without cleanup, the median moves 5.56 m → 7.03 m,
which coincides with the IRC:86 two-lane design width of 7.0 m.

**Adopted:** the two stages use different masks, because they ask different
questions. The vanishing-point stage asks where the road converges and is best
served by the clean unoccluded fragment. The width stage asks how wide the
carriageway is and must see through the traffic standing on it.

**Consequence for the scale calibration.** The carriageway anchor in
`scale_calibration.md` was measured on unbridged masks and so was biased low by
occlusion as well as by frame clipping. Corrected, it supports the adopted
camera height of 1.75 m rather than contradicting it, and now agrees with the
vehicle-width anchor instead of disagreeing with it.
