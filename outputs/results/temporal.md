# Temporal stability — IDD Temporal Val clips

150 sequences, 4526 frames.

Stability, not accuracy: no dataset provides ground-truth lane geometry
for Indian roads. Jitter is the median absolute frame-to-frame change;
over a three-second clip the road ahead barely changes, so large swings
are estimator noise. Solution rate and empirical-override rate are shown
alongside, because a filter that ignored its input would score perfectly
on jitter alone.

Statistics pool every frame-to-frame transition across all clips.

| Configuration | Solution rate | Width jitter (p50 / p90) | Offset jitter | Heading jitter | Lane-count flips /100 | Empirical override |
|---|---|---|---|---|---|---|
| per-frame (no temporal) | 80.4% | 0.311 / 2.345 m | 0.602 m | 0.0866 | 20.14 | 0.0% |
| + Kalman + count vote | 85.4% | 0.038 / 0.173 m | 0.148 m | 0.0191 | 4.28 | 0.0% |
| + empirical lanes from traffic | 85.4% | 0.039 / 0.184 m | 0.148 m | 0.0191 | 4.97 | 4.7% |