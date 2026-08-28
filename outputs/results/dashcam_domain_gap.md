# Generalising to consumer dashcam footage

A user-supplied frame from a 70mai dashcam — dusk, wide-angle lens with heavy
vignetting, the vehicle's own bonnet across the bottom of the frame — produced a
visibly poor result. This records what was tried, and what worked.

## Two genuine defects, both fixed

These were real bugs, not domain gap, and the frame exposed them:

**Bottom-connectivity assumed the road reaches the last row.** True on IDD, false
on any dashcam that sees its own bonnet. The road stopped above the bonnet and
was discarded as "not touching the bottom", while a 972-pixel false positive *on*
the bonnet was kept — 48,461 px of correctly-segmented carriageway thrown away.
The rule is now "low in the frame and large" rather than "touching the last row".
IDD improved as a side effect, 0.9067 → 0.9258.

**A degenerate carriageway width.** `infer_lanes` measures the carriageway twice:
a median over the profile, and the two fitted edge polynomials evaluated at a
reference distance. Only the first was bounds-checked. Where the fits cross, the
second collapsed to its `1e-6` clamp and a zero-width carriageway propagated
through every downstream stage. On this frame: 0.00 m → 9.03 m, three lanes,
which matches the road.

## The remaining gap is the training distribution

IDD is daylight footage from a normal lens. Nothing in it resembles a wide-angle
sensor at dusk. Four approaches were measured.

| Approach | Effect on the dashcam frame | Effect on IDD |
|---|---|---|
| CLAHE applied unconditionally | drivable 13.0% → 29.7% | **IoU 0.9437 → 0.8790** |
| Adaptive CLAHE, triggered by image statistics | — | no trigger exists (below) |
| Dual hypothesis, select by lane confidence | picks the wrong one | IoU 0.9329 → 0.9062 |
| Dual hypothesis, select by classical agreement | picks correctly | IoU 0.9402 → 0.9002 |
| Fine-tune with synthetic dashcam augmentation | 13.6% → **13.5%** | 0.9329 → 0.9303 |

**Contrast enhancement works and cannot be applied.** CLAHE more than doubles
recovered drivable area on this frame and cuts the spurious sky prediction from
41.5% to 27.5%, but costs 6.5 points of IoU on IDD. It is only correct on frames
that need it.

**No image statistic identifies those frames.** Lightness standard deviation,
percentile spread, median and dark-pixel fraction were all measured across the
validation split against the dashcam frame. It sits *inside* IDD's normal range
on every one of them:

| statistic | IDD p05 | IDD p50 | IDD p95 | dashcam |
|---|---|---|---|---|
| L std | 49.5 | 66.4 | 86.9 | 53.7 |
| L spread (p95−p5) | 152.7 | 215.5 | 248.7 | 178.0 |
| L median | 50.0 | 92.5 | 126.2 | 87.0 |
| dark fraction | 0.1 | 0.2 | 0.4 | 0.2 |

So the failure is not "low contrast". CLAHE helps for a different reason —
breaking up a large, smoothly-varying hazy region that the network otherwise
labels sky.

**Prediction-side triggers are too weak.** Sky predicted below the horizon is
physically impossible and was tested as a trigger: 0.47% on the dashcam against a
maximum of 0.43% across the validation split. It does separate, but by a margin
far too narrow to gate on.

**Fine-tuning on synthetic dashcam conditions did not transfer.** Synthetic
vignetting, non-linear low-light response with sensor noise, and barrel
distortion were added to the augmentation suite, and the trained model was
fine-tuned for 12 epochs. Recovered drivable area on the real frame moved from
13.6% to 13.5% — no change — while IDD lost 0.0026 IoU. Twelve epochs from an
already-converged model may simply have re-converged, but the more likely reading
is that the gap is in sensor, lens and ISP characteristics that a synthetic
brightness-and-geometry transform does not reproduce.

## Conclusion

`seg_idd20k` remains the shipped checkpoint. The fine-tuned variant is retained
under `seg_robust` but is not used, because it costs accuracy on the evaluation
set and buys nothing on the case it was built for.

The honest statement of the limitation: **the system is trained on daylight,
normal-lens footage and degrades on wide-angle sensors at dusk.** The fix is
training data of that kind, not a test-time transform. IDD-AW — the adverse
weather subset, which includes low-light captures — is the obvious next source,
and is the one substantive experiment this analysis leaves open.
