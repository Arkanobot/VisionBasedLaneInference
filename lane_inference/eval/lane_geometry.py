"""Accuracy of the inferred carriageway, measured against ground truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config as cfg


def reference_edges(gt_mask_bev: np.ndarray, ipm, Ys, footprint):
    """Left/right extent of the ground-truth road at each forward distance."""
    bh, bw = gt_mask_bev.shape[:2]
    binary = gt_mask_bev > 0
    inside = footprint > 0
    out = {}
    for Y in Ys:
        row = int(round(ipm.ground_to_bev([[0.0, float(Y)]])[0][1]))
        if not (0 <= row < bh):
            continue
        xs = np.flatnonzero(binary[row])
        if xs.size < 3:
            continue
        breaks = np.flatnonzero(np.diff(xs) > 1)
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks, [xs.size - 1]])
        k = int(np.argmax(ends - starts))
        a, b = int(xs[starts[k]]), int(xs[ends[k]])
        if b - a < 3:
            continue

        if not (a > 1 and b < bw - 2):
            continue
        if not (a - 1 >= 0 and inside[row, a - 1] and b + 1 < bw and inside[row, b + 1]):
            continue
        g = ipm.bev_to_ground([[a, row], [b, row]])
        out[float(Y)] = (float(g[0][0]), float(g[1][0]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Carriageway accuracy vs ground truth")
    p.add_argument("--split", default="val")
    p.add_argument("--data-root", default=str(cfg.DATA_ROOT))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=str(cfg.OUTPUT_ROOT / "results"))
    args = p.parse_args()

    from core.dl_drivable import SegmenterBackend
    from core.lane_tracker import bev_footprint
    from core.pipeline import LaneInferencePipeline, PipelineOptions
    from data.idd import IDDLite

    ds = IDDLite(args.data_root, args.split, input_size=(227, 320), augment=False)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))

    backend = SegmenterBackend(args.checkpoint)
    print(f"model: {backend.checkpoint_path.name}\n")
    pipe = LaneInferencePipeline(backend=backend,
                                 options=PipelineOptions(use_dl=True, use_cv=True))

    Ys = np.arange(6.0, 22.1, 1.0)
    left_err, right_err, width_err, rel_err, centre_err = [], [], [], [], []
    per_frame = []
    compared = solved = 0

    for i in range(n):
        img, gt = ds.raw(i)
        r = pipe.process(img)
        if r.lane_model is None or r.ipm is None:
            continue
        solved += 1

        gt_bev = r.ipm.warp_mask((gt == cfg.DRIVABLE_ID).astype(np.uint8) * 255)
        ref = reference_edges(gt_bev, r.ipm, Ys, bev_footprint(r.ipm))
        if len(ref) < 4:
            continue
        compared += 1
        frame_err = []

        for Y, (gl, gr) in ref.items():
            b = r.lane_model.boundaries_at(Y)
            if len(b) < 2:
                continue
            il, ir = float(b[0]), float(b[-1])
            gw = gr - gl
            if gw <= 0.5:
                continue
            left_err.append(abs(il - gl))
            right_err.append(abs(ir - gr))
            width_err.append(abs((ir - il) - gw))
            rel_err.append(abs((ir - il) - gw) / gw)
            centre_err.append(abs(0.5 * (il + ir) - 0.5 * (gl + gr)))
            frame_err.append(0.5 * (abs(il - gl) + abs(ir - gr)))

        if frame_err:
            per_frame.append({"confidence": float(r.lane_model.confidence),
                              "error_m": float(np.median(frame_err))})

    if not left_err:
        raise SystemExit("no comparable samples")

    def st(a):
        a = np.asarray(a)
        return dict(median=float(np.median(a)), mean=float(a.mean()),
                    p90=float(np.percentile(a, 90)))

    res = {
        "frames": n, "with_lane_model": solved, "compared": compared,
        "samples": len(left_err),
        "left_edge_error_m": st(left_err),
        "right_edge_error_m": st(right_err),
        "width_error_m": st(width_err),
        "width_error_relative": st(rel_err),
        "centre_error_m": st(centre_err),
        "camera_height_m": cfg.DEFAULT.geometry.camera_height_m,
    }


    if per_frame:
        conf = np.array([f["confidence"] for f in per_frame])
        errs = np.array([f["error_m"] for f in per_frame])
        gates = []
        for g in (0.0, 0.6, 0.7, 0.8):
            keep = conf >= g
            if keep.sum():
                gates.append({"gate": g, "kept": int(keep.sum()),
                              "kept_pct": float(100 * keep.mean()),
                              "median_error_m": float(np.median(errs[keep])),
                              "p90_error_m": float(np.percentile(errs[keep], 90))})
        res["confidence_gates"] = gates
        res["confidence_error_correlation"] = float(np.corrcoef(conf, errs)[0, 1])
        res["frames_scored"] = len(per_frame)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "lane_geometry.json").write_text(json.dumps(res, indent=2))

    print(f"frames {n}, lane model on {solved}, comparable {compared}, "
          f"{len(left_err)} boundary samples over 6-22 m\n")
    print(f"{'quantity':<26}{'median':>9}{'mean':>9}{'p90':>9}")
    print("-" * 53)
    for k, label in [("left_edge_error_m", "left edge error"),
                     ("right_edge_error_m", "right edge error"),
                     ("centre_error_m", "centreline error"),
                     ("width_error_m", "carriageway width error")]:
        v = res[k]
        print(f"{label:<26}{v['median']:>9.3f}{v['mean']:>9.3f}{v['p90']:>9.3f}   m")
    v = res["width_error_relative"]
    print(f"{'width error, relative':<26}{v['median']*100:>8.1f}%{v['mean']*100:>8.1f}%"
          f"{v['p90']*100:>8.1f}%")
    print(f"\nabsolute errors scale with the assumed camera height "
          f"({res['camera_height_m']} m); the relative figure does not.")


if __name__ == "__main__":
    main()
