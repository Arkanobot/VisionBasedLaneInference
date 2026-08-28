"""Where the pipeline fails, and why."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import config as cfg


def diagnose(result, pipe) -> str:
    """Attribute a missing lane solution to a specific stage."""
    if result.scene is not None and not result.scene:
        return "scene rejected"
    if result.ipm is None:
        return "no rectification"
    from core.geometry import vp_from_road_mask
    from core.lane_tracker import extract_road_profile
    if vp_from_road_mask(result.road_mask) is None:
        return "no vanishing point"
    prof = extract_road_profile(result.ipm.warp_mask(result.road_mask), result.ipm)
    if prof.left_valid.sum() < 4 or prof.right_valid.sum() < 4:
        return "road edges not visible"
    w = prof.width[np.isfinite(prof.width)]
    if len(w) and np.median(w) < cfg.DEFAULT.geometry.min_lane_width_m:
        return "carriageway too narrow"
    return "other"


def main() -> None:
    p = argparse.ArgumentParser(description="Failure analysis")
    p.add_argument("--split", default="val")
    p.add_argument("--data-root", default=str(cfg.DATA_ROOT))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--top", type=int, default=12, help="worst frames to render")
    p.add_argument("--out", default=str(cfg.OUTPUT_ROOT / "results"))
    args = p.parse_args()

    from core.dl_drivable import SegmenterBackend
    from core.lane_tracker import bev_footprint
    from core.pipeline import LaneInferencePipeline, PipelineOptions
    from core.visualizer import draw_lane_overlay, stack_panels
    from data.idd import IDDLite
    from eval.lane_geometry import reference_edges
    from utils.image_utils import overlay_mask

    ds = IDDLite(args.data_root, args.split, input_size=(227, 320), augment=False)
    backend = SegmenterBackend(args.checkpoint)
    pipe = LaneInferencePipeline(backend=backend,
                                 options=PipelineOptions(use_dl=True, use_cv=True))
    Ys = np.arange(6.0, 22.1, 1.0)

    causes: dict[str, int] = {}
    scored, unsolved = [], []

    for i in range(len(ds)):
        img, gt = ds.raw(i)
        r = pipe.process(img)
        veh = float((gt == cfg.VEHICLE_ID).mean())
        road = float((gt == cfg.DRIVABLE_ID).mean())

        if r.lane_model is None:
            why = diagnose(r, pipe)
            causes[why] = causes.get(why, 0) + 1
            unsolved.append({"idx": i, "cause": why, "vehicle_frac": veh,
                             "road_frac": road})
            continue

        ref = reference_edges(r.ipm.warp_mask((gt == cfg.DRIVABLE_ID).astype(np.uint8)*255),
                              r.ipm, Ys, bev_footprint(r.ipm))
        if len(ref) < 4:
            continue
        errs = [0.5*(abs(r.lane_model.boundaries_at(Y)[0]-gl)
                     + abs(r.lane_model.boundaries_at(Y)[-1]-gr))
                for Y, (gl, gr) in ref.items() if len(r.lane_model.boundaries_at(Y)) >= 2]
        if not errs:
            continue
        scored.append({"idx": i, "error": float(np.median(errs)),
                       "confidence": float(r.lane_model.confidence),
                       "lanes": int(r.lane_model.n_lanes),
                       "width": float(r.lane_model.carriageway_width_m),
                       "vehicle_frac": veh, "road_frac": road})

    scored.sort(key=lambda d: -d["error"])
    err = np.array([d["error"] for d in scored])
    worst = scored[:max(len(scored) // 5, 1)]
    best = scored[-max(len(scored) // 5, 1):]

    def avg(rows, k):
        return float(np.mean([r[k] for r in rows])) if rows else float("nan")

    summary = {
        "frames": len(ds),
        "solved": len(scored) + 0,
        "unsolved": len(unsolved),
        "causes": dict(sorted(causes.items(), key=lambda kv: -kv[1])),
        "error_median": float(np.median(err)),
        "error_p90": float(np.percentile(err, 90)),
        "worst_quintile": {"error": avg(worst, "error"),
                           "confidence": avg(worst, "confidence"),
                           "vehicle_frac": avg(worst, "vehicle_frac"),
                           "road_frac": avg(worst, "road_frac")},
        "best_quintile": {"error": avg(best, "error"),
                          "confidence": avg(best, "confidence"),
                          "vehicle_frac": avg(best, "vehicle_frac"),
                          "road_frac": avg(best, "road_frac")},
        "unsolved_vehicle_frac": avg(unsolved, "vehicle_frac"),
        "solved_vehicle_frac": avg(scored, "vehicle_frac"),
    }

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "failures.json").write_text(json.dumps(
        {"summary": summary, "worst": scored[:args.top], "unsolved": unsolved}, indent=2))


    tiles = []
    for d in scored[:args.top // 2] + unsolved[:args.top // 2]:
        img, gt = ds.raw(d["idx"])
        r = pipe.process(img)
        ov = (draw_lane_overlay(img, r.lane_model, r.ipm, y_max=22.0)
              if r.lane_model else img.copy())
        ref_img = overlay_mask(img, (gt == cfg.DRIVABLE_ID).astype(np.uint8)*255,
                               (60, 60, 220), 0.45)
        tag = (f"#{d['idx']} err {d['error']:.2f}m conf {d['confidence']:.2f}"
               if "error" in d else f"#{d['idx']} {d['cause']}")
        tiles.append(stack_panels([img, ref_img, ov], height=190,
                                  labels=[tag, "ground truth road", "inferred"]))
    if tiles:
        W = max(t.shape[1] for t in tiles)
        grid = np.concatenate([np.pad(t, ((0, 4), (0, W - t.shape[1]), (0, 0)))
                               for t in tiles], axis=0)
        cv2.imwrite(str(cfg.OUTPUT_ROOT / "failure_gallery.png"), grid)


    s = summary
    lines = [
        "# Failure analysis", "",
        f"IDD-Lite {args.split}, {s['frames']} frames. A lane solution was produced "
        f"on {s['frames']-s['unsolved']} and refused on {s['unsolved']}.", "",
        "## Why frames produce no solution", "",
        "| Cause | Frames |", "|---|---|",
    ]
    for k, v in s["causes"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "Refusing is the correct behaviour where the carriageway is not "
                  "visible; a wrong lane model is worse than none.", "",
              "## What distinguishes accurate frames from inaccurate ones", "",
              "Frames with a solution, split into best and worst fifths by "
              "boundary error:", "",
              "| | best fifth | worst fifth |", "|---|---|---|",
              f"| Boundary error | {s['best_quintile']['error']:.3f} m | "
              f"{s['worst_quintile']['error']:.3f} m |",
              f"| Reported confidence | {s['best_quintile']['confidence']:.2f} | "
              f"{s['worst_quintile']['confidence']:.2f} |",
              f"| Vehicle pixels | {s['best_quintile']['vehicle_frac']*100:.1f}% | "
              f"{s['worst_quintile']['vehicle_frac']*100:.1f}% |",
              f"| Road pixels | {s['best_quintile']['road_frac']*100:.1f}% | "
              f"{s['worst_quintile']['road_frac']*100:.1f}% |", "",
              f"Frames with no solution average {s['unsolved_vehicle_frac']*100:.1f}% "
              f"vehicle pixels against {s['solved_vehicle_frac']*100:.1f}% for solved "
              f"frames.", "",
              "Rendered examples: `outputs/failure_gallery.png`."]
    md = "\n".join(lines)
    (out / "failures.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
