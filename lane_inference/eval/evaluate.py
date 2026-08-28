"""Quantitative evaluation and ablation study on IDD-Lite."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import config as cfg
from data.idd import IDDLite
from eval.metrics import (MARKDOWN_HEADER, BoundaryAccumulator, ConfusionMatrix,
                          format_table, markdown_row)


class Method:
    """A named configuration returning a drivable mask, optionally full labels."""

    def __init__(self, key, name, fn, multiclass=False, note=""):
        self.key, self.name, self.fn = key, name, fn
        self.multiclass, self.note = multiclass, note

    def __call__(self, image):
        return self.fn(image)


def build_methods(args) -> list[Method]:
    from core.asphalt_detector import detect_asphalt
    from eval.baselines import (bottom_trapezoid_prior, const_all_road,
                                phase3_cv_only, phase3_hybrid)

    methods: list[Method] = []

    if not args.skip_trivial:
        methods += [
            Method("all_road", "trivial: everything is road",
                   lambda im: (const_all_road(im), None)),
            Method("trapezoid", "trivial: fixed trapezoid prior",
                   lambda im: (bottom_trapezoid_prior(im), None)),
        ]

    methods.append(Method("phase3_cv", "Phase 3: HSV threshold (CV only)",
                          lambda im: (phase3_cv_only(im), None)))


    yolop = legacy_yolop = None
    if not args.skip_yolop:
        try:
            from core.dl_drivable import LegacyYolopBackend, YolopBackend
            print("loading YOLOP ...", flush=True)
            legacy_yolop = LegacyYolopBackend()
            yolop = YolopBackend()
        except Exception as e:                       # pragma: no cover
            print(f"  YOLOP unavailable, skipping ({type(e).__name__}: {e})")

    if legacy_yolop is not None:
        def _legacy_da(im):
            da, _, _ = legacy_yolop(im)
            return da, None

        def _phase3_hybrid(im):
            da, _, _ = legacy_yolop(im)
            return phase3_hybrid(im, da), None

        methods += [
            Method("yolop_legacy", "Phase 3: YOLOP as wired in Phase 3", _legacy_da,
                   note="stretched resize, no ImageNet normalisation, CPU, 2 threads"),
            Method("phase3_hybrid", "Phase 3: HSV OR YOLOP (the Phase 3 pipeline)",
                   _phase3_hybrid),
        ]
    if yolop is not None:
        methods.append(Method("yolop_fixed", "YOLOP, correctly wired",
                              lambda im: (yolop(im)[0], None),
                              note="letterboxed + ImageNet-normalised"))


        def _yolop_fused(im):
            from core.dl_drivable import fuse_masks
            da, _, _ = yolop(im)
            _, cv_conf = detect_asphalt(im, return_confidence=True)
            mask, _ = fuse_masks(cv_conf, (da > 0).astype(np.float32))
            return mask, None

        methods.append(Method("yolop_fused",
                              "YOLOP (correct) + confidence-weighted CV fusion",
                              _yolop_fused,
                              note="tests fusion on an out-of-domain learned branch"))

    methods.append(Method("phase4_cv", "Phase 4: adaptive CIELAB + texture (CV only)",
                          lambda im: (detect_asphalt(im), None)))


    backend = None
    ckpt = Path(args.checkpoint) if args.checkpoint else cfg.default_checkpoint()
    if ckpt.is_file():
        from core.dl_drivable import SegmenterBackend
        from core.pipeline import LaneInferencePipeline, PipelineOptions
        backend = SegmenterBackend(str(ckpt))
        print(f"loaded {ckpt} (training mIoU {backend.trained_miou:.4f})")

        def _phase4_dl(im):


            r = backend(im)
            return r.drivable, r.labels

        methods.append(Method("phase4_dl", "Phase 4: IDD-trained segmentation",
                              _phase4_dl, multiclass=True))

        for key, name, opt in [
            ("phase4_dl_clean", "Phase 4: learned + cleanup (no CV fusion)",
             PipelineOptions(use_dl=True, use_cv=False, use_fusion=False,
                             use_cleanup=True, use_ipm=False)),
            ("phase4_fusion", "Phase 4: learned + confidence-weighted CV fusion",
             PipelineOptions(use_dl=True, use_cv=True, use_fusion=True,
                             use_cleanup=False, use_ipm=False)),
            ("phase4_full", "Phase 4: learned + CV fusion + cleanup",
             PipelineOptions(use_dl=True, use_cv=True, use_fusion=True,
                             use_cleanup=True, use_ipm=False)),
        ]:
            pipe = LaneInferencePipeline(backend=backend, options=opt)
            methods.append(Method(key, name,
                                  lambda im, p=pipe: (p.process(im).road_mask, None)))
    else:
        print(f"no checkpoint at {ckpt} - learned configurations skipped.\n"
              f"Train one, or copy seg_mnv3/best.pt from the Colab run.")

    if args.methods:
        wanted = {m.strip() for m in args.methods.split(",")}
        methods = [m for m in methods if m.key in wanted]
    return methods


def evaluate_method(method: Method, dataset: IDDLite, limit: int | None = None,
                    boundary_tol: int = 3) -> dict:
    n = len(dataset) if limit is None else min(limit, len(dataset))
    binary_cm = ConfusionMatrix(num_classes=2)
    full_cm = ConfusionMatrix(num_classes=cfg.NUM_CLASSES) if method.multiclass else None
    boundary = BoundaryAccumulator(boundary_tol)
    latencies = []


    if n:
        method(dataset.raw(0)[0])

    for i in range(n):
        image, gt = dataset.raw(i)
        t0 = time.perf_counter()
        mask, labels = method(image)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        pred_bool = mask > 0
        gt_bool = (gt == cfg.DRIVABLE_ID)
        valid = (gt != cfg.IGNORE_INDEX)

        binary_cm.update(pred_bool[valid].astype(np.uint8),
                         gt_bool[valid].astype(np.uint8))
        boundary.update(pred_bool, gt_bool)
        if full_cm is not None and labels is not None:
            full_cm.update(labels, gt)

    s = binary_cm.summary()
    result = {
        "key": method.key,
        "name": method.name,
        "note": method.note,
        "frames": n,

        "drivable_iou": float(s["per_class_iou"][1]),
        "drivable_precision": float(s["per_class_precision"][1]),
        "drivable_recall": float(s["per_class_recall"][1]),
        "drivable_dice": float(s["per_class_dice"][1]),
        "pixel_accuracy": s["pixel_accuracy"],
        "latency_ms_median": float(np.median(latencies)),
        "latency_ms_mean": float(np.mean(latencies)),
        "fps": float(1000.0 / max(np.median(latencies), 1e-6)),
        **boundary.score(),
    }
    if full_cm is not None:
        fs = full_cm.summary()
        result["miou"] = fs["miou"]
        result["per_class_iou"] = [float(v) for v in fs["per_class_iou"]]
        result["_full_summary"] = fs
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="IDD-Lite evaluation / ablation")
    p.add_argument("--split", default="val")
    p.add_argument("--data-root", default=str(cfg.DATA_ROOT))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--methods", default=None, help="comma-separated method keys")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-yolop", action="store_true")
    p.add_argument("--skip-trivial", action="store_true")
    p.add_argument("--boundary-tol", type=int, default=3)
    p.add_argument("--out", default=str(cfg.OUTPUT_ROOT / "results"))
    args = p.parse_args()

    ds = IDDLite(args.data_root, args.split, input_size=(227, 320), augment=False)
    methods = build_methods(args)
    if not methods:
        print("no methods selected")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"\nevaluating {len(methods)} configuration(s) on {n} {args.split} frames\n")

    results = []
    for m in methods:
        t0 = time.time()
        r = evaluate_method(m, ds, args.limit, args.boundary_tol)
        results.append(r)
        print(f"{m.name:<48} IoU {r['drivable_iou']:.4f}  "
              f"BF1 {r['boundary_f1']:.4f}  {r['fps']:7.1f} FPS   "
              f"[{time.time()-t0:.0f}s]", flush=True)


    lines = [f"# Drivable-area evaluation - IDD-Lite {args.split} split ({n} frames)", ""]
    lines += [MARKDOWN_HEADER.replace("| mIoU ", "| mIoU ")]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['drivable_iou']:.4f} | {r['drivable_precision']:.4f} "
            f"| {r['drivable_recall']:.4f} | "
            f"{r.get('miou', float('nan')):.4f} | {r['boundary_f1']:.4f} "
            f"| {r['fps']:.1f} |")
    notes = [r for r in results if r["note"]]
    if notes:
        lines += ["", "Notes:"] + [f"- **{r['name']}**: {r['note']}" for r in notes]

    for r in results:
        if "_full_summary" in r:
            lines += ["", f"## Per-class detail - {r['name']}", "",
                      "```", format_table(r["_full_summary"]), "```"]
            del r["_full_summary"]

    md = "\n".join(lines)
    (out_dir / f"evaluation_{args.split}.md").write_text(md)
    (out_dir / f"evaluation_{args.split}.json").write_text(json.dumps(results, indent=2))
    print("\n" + md)
    print(f"\nwritten to {out_dir}/evaluation_{args.split}.{{md,json}}")


if __name__ == "__main__":
    main()
