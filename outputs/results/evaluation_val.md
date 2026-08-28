# Drivable-area evaluation - IDD-Lite val split (204 frames)

| Configuration | Drivable IoU | Precision | Recall | mIoU | Boundary F1 | FPS |
|---|---|---|---|---|---|---|
| trivial: everything is road | 0.3179 | 0.3179 | 1.0000 | nan | nan | 428468.4 |
| trivial: fixed trapezoid prior | 0.6450 | 0.8246 | 0.7476 | nan | 0.1957 | 205127.9 |
| Phase 3: HSV threshold (CV only) | 0.4131 | 0.4682 | 0.7783 | nan | 0.2093 | 2336.2 |
| Phase 3: YOLOP as wired in Phase 3 | 0.1667 | 0.9119 | 0.1694 | nan | 0.1382 | 6.4 |
| Phase 3: HSV OR YOLOP (the Phase 3 pipeline) | 0.4270 | 0.4747 | 0.8093 | nan | 0.2048 | 6.3 |
| YOLOP, correctly wired | 0.7243 | 0.9923 | 0.7284 | nan | 0.2205 | 6.2 |
| YOLOP (correct) + confidence-weighted CV fusion | 0.6979 | 0.9181 | 0.7443 | nan | 0.2324 | 5.8 |
| Phase 4: adaptive CIELAB + texture (CV only) | 0.5697 | 0.7709 | 0.6858 | nan | 0.2099 | 361.0 |
| Phase 4: IDD-trained segmentation | 0.9403 | 0.9821 | 0.9568 | 0.7133 | 0.8343 | 67.5 |
| Phase 4: learned + cleanup (no CV fusion) | 0.9310 | 0.9855 | 0.9440 | nan | 0.7754 | 58.0 |
| Phase 4: learned + confidence-weighted CV fusion | 0.9066 | 0.9719 | 0.9310 | nan | 0.7234 | 50.7 |
| Phase 4: learned + CV fusion + cleanup | 0.9047 | 0.9781 | 0.9234 | nan | 0.7031 | 50.3 |

Notes:
- **Phase 3: YOLOP as wired in Phase 3**: stretched resize, no ImageNet normalisation, CPU, 2 threads
- **YOLOP, correctly wired**: letterboxed + ImageNet-normalised
- **YOLOP (correct) + confidence-weighted CV fusion**: tests fusion on an out-of-domain learned branch

## Per-class detail - Phase 4: IDD-trained segmentation

```
class                   IoU     Dice     Prec   Recall
------------------------------------------------------
drivable             0.9403   0.9692   0.9821   0.9568
non-drivable         0.4479   0.6187   0.5302   0.7428
living-things        0.5666   0.7234   0.6682   0.7886
vehicles             0.7865   0.8805   0.8774   0.8837
roadside-objects     0.5377   0.6994   0.7028   0.6960
far-objects          0.7650   0.8668   0.8728   0.8610
sky                  0.9487   0.9737   0.9751   0.9722
------------------------------------------------------
mean                 0.7133   0.8188
pixel accuracy   : 0.8920
mean class acc.  : 0.8430
```