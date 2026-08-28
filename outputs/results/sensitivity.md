# Lane inference sensitivity to the metric calibration

Camera height is the pipeline's only metric free parameter, and
lateral scale is exactly proportional to it.

| camera height | frames solved | median carriageway | median lane width | mean lane count | lane-count distribution |
|---|---|---|---|---|---|
| 1.30 m | 146/204 | 4.67 m | 3.41 m | 1.58 | 1:85, 2:48, 3:7, 4:4, 5:1, 8:1 |
| 1.50 m | 156/204 | 4.97 m | 3.49 m | 1.60 | 1:80, 2:64, 3:9, 4:2, 6:1 |
| 1.75 m | 165/204 | 5.27 m | 3.41 m | 1.68 | 1:73, 2:75, 3:14, 4:2, 5:1 |
| 2.00 m | 171/204 | 5.90 m | 3.51 m | 1.86 | 1:58, 2:90, 3:18, 4:3, 6:1, 8:1 |
| 2.40 m | 166/204 | 6.98 m | 3.54 m | 2.04 | 1:37, 2:95, 3:28, 4:4, 6:2 |

## Reading this table

**Coverage rises with assumed height.** A solution is rejected when the
recovered carriageway falls below `min_lane_width_m` (2.5 m), and a larger
assumed height scales every width up, so fewer frames are rejected. Coverage
is therefore *not* independent of the calibration, and a high coverage number
at a large assumed height is not evidence that the height is correct.

**Median lane width is remarkably stable** across the whole sweep, because
the lane count is an integer division: as the carriageway grows, the inferred
count grows with it and the quotient stays near the IRC nominal. This is a
genuine robustness property of partitioning rather than thresholding - the
*structure* the pipeline reports degrades gracefully under scale error even
though the absolute widths do not.

**The lane count is what actually moves.** Between the extremes of the sweep
the modal road shifts from single-lane to two-lane. Lane counts should
therefore be read as conditional on the stated camera height, and a
deployment that needs them exactly should fix the scale with
`IPMTransform.calibrate_from_known_width()`.