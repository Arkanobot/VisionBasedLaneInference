# Failure analysis

IDD-Lite val, 204 frames. A lane solution was produced on 165 and refused on 39.

## Why frames produce no solution

| Cause | Frames |
|---|---|
| carriageway too narrow | 20 |
| road edges not visible | 11 |
| no vanishing point | 8 |

Refusing is the correct behaviour where the carriageway is not visible; a wrong lane model is worse than none.

## What distinguishes accurate frames from inaccurate ones

Frames with a solution, split into best and worst fifths by boundary error:

| | best fifth | worst fifth |
|---|---|---|
| Boundary error | 0.181 m | 1.199 m |
| Reported confidence | 0.79 | 0.68 |
| Vehicle pixels | 10.7% | 9.2% |
| Road pixels | 31.1% | 32.8% |

Frames with no solution average 10.2% vehicle pixels against 8.2% for solved frames.

Rendered examples: `outputs/failure_gallery.png`.