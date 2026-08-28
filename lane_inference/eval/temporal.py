"""Temporal evaluation on IDD Temporal video sequences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import config as cfg


def run_clip(path: Path, pipe, max_frames: int | None = None) -> dict:
    """Run one clip and collect per-frame lane parameters."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {}
    pipe.reset()

    widths, offsets, headings, counts, sources = [], [], [], [], []
    solved = total = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and total >= max_frames):
            break
        total += 1
        r = pipe.process(frame)
        m = r.lane_model
        if m is None or m.left_coeffs is None:
            continue
        solved += 1
        centre = 0.5 * (m.left_coeffs + m.right_coeffs)
        widths.append(m.carriageway_width_m)
        offsets.append(float(centre[2]))
        headings.append(float(centre[1]))
        counts.append(int(m.n_lanes))
        sources.append(r.lane_source)
    cap.release()

    if solved < 4:
        return {}

    def diffs(v):
        return np.abs(np.diff(np.asarray(v, dtype=float)))

    def jitter(v):
        d = diffs(v)
        return float(np.median(d)) if len(d) else float("nan")

    flips = int(np.sum(np.asarray(counts[1:]) != np.asarray(counts[:-1])))
    return {
        "clip": path.name,


        "_width_diffs": diffs(widths).tolist(),
        "_offset_diffs": diffs(offsets).tolist(),
        "_heading_diffs": diffs(headings).tolist(),
        "_flips": flips,
        "_transitions": max(len(counts) - 1, 0),
        "frames": total,
        "solved": solved,
        "solution_rate": solved / max(total, 1),
        "width_jitter_m": jitter(widths),
        "offset_jitter_m": jitter(offsets),
        "heading_jitter": jitter(headings),
        "count_flips_per_100": 100.0 * flips / max(len(counts) - 1, 1),
        "modal_lane_count": int(np.bincount(counts).argmax()) if counts else -1,
        "empirical_frac": float(np.mean([s == "empirical" for s in sources])),
        "median_width_m": float(np.median(widths)),
    }


def aggregate(rows: list[dict]) -> dict:
    """Pool every frame-to-frame transition across all clips before taking the"""
    if not rows:
        return {}
    pool = lambda key: np.concatenate([np.asarray(r[key]) for r in rows
                                       if len(r[key])]) if rows else np.empty(0)
    wd, od, hd = pool("_width_diffs"), pool("_offset_diffs"), pool("_heading_diffs")
    flips = sum(r["_flips"] for r in rows)
    trans = sum(r["_transitions"] for r in rows)
    solved = sum(r["solved"] for r in rows)
    frames = sum(r["frames"] for r in rows)

    med = lambda a: float(np.median(a)) if len(a) else float("nan")
    p90 = lambda a: float(np.percentile(a, 90)) if len(a) else float("nan")
    return {
        "clips": len(rows),
        "frames": int(frames),
        "solution_rate": solved / max(frames, 1),
        "width_jitter_m": med(wd),
        "width_jitter_p90_m": p90(wd),
        "offset_jitter_m": med(od),
        "offset_jitter_p90_m": p90(od),
        "heading_jitter": med(hd),
        "count_flips_per_100": 100.0 * flips / max(trans, 1),
        "empirical_frac": float(np.mean([r["empirical_frac"] for r in rows])),
        "transitions": int(trans),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Temporal stability evaluation")
    p.add_argument("--clips", default=str(cfg.ROOT / "datasets" / "idd_clips"))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--limit", type=int, default=None, help="max clips")
    p.add_argument("--max-frames", type=int, default=None, help="max frames per clip")
    p.add_argument("--out", default=str(cfg.OUTPUT_ROOT / "results"))
    args = p.parse_args()

    clip_dir = Path(args.clips)
    clips = sorted(clip_dir.glob("*.mp4"))
    if args.limit:
        clips = clips[:args.limit]
    if not clips:
        raise SystemExit(f"no .mp4 clips under {clip_dir}. Run the temporal Colab "
                         f"notebook and unpack idd_clips.tar.gz there.")

    from core.dl_drivable import SegmenterBackend
    from core.pipeline import LaneInferencePipeline, PipelineOptions

    backend = None
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    try:
        backend = SegmenterBackend(str(ckpt) if ckpt else None)
        print(f"model: {backend.checkpoint_path.name} on {backend.device}")
    except FileNotFoundError as e:
        print(f"{e}\nfalling back to the classical backend")

    base = dict(use_dl=backend is not None, use_cv=True)
    configs = {
        "per-frame (no temporal)":
            PipelineOptions(**base, use_temporal=False, use_trajectory_lanes=False),
        "+ Kalman + count vote":
            PipelineOptions(**base, use_temporal=True, use_trajectory_lanes=False),
        "+ empirical lanes from traffic":
            PipelineOptions(**base, use_temporal=True, use_trajectory_lanes=True),
    }

    print(f"\n{len(clips)} clips from {clip_dir}\n")
    results = {}
    for name, opt in configs.items():
        pipe = LaneInferencePipeline(backend=backend, options=opt)
        rows = []
        for i, c in enumerate(clips):
            r = run_clip(c, pipe, args.max_frames)
            if r:
                rows.append(r)
            if (i + 1) % 25 == 0:
                print(f"  {name}: {i+1}/{len(clips)} clips", flush=True)
        summary = aggregate(rows)
        slim = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        results[name] = {"summary": summary, "per_clip": slim}
        s = results[name]["summary"]
        print(f"{name:<32} solved {s['solution_rate']*100:5.1f}%  "
              f"width jitter {s['width_jitter_m']:.3f} m  "
              f"count flips {s['count_flips_per_100']:5.2f}/100", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "temporal.json").write_text(json.dumps(results, indent=2))

    lines = ["# Temporal stability — IDD Temporal Val clips", "",
             f"{len(clips)} sequences, "
             f"{results[list(configs)[0]]['summary'].get('frames', 0)} frames.", "",
             "Stability, not accuracy: no dataset provides ground-truth lane geometry",
             "for Indian roads. Jitter is the median absolute frame-to-frame change;",
             "over a three-second clip the road ahead barely changes, so large swings",
             "are estimator noise. Solution rate and empirical-override rate are shown",
             "alongside, because a filter that ignored its input would score perfectly",
             "on jitter alone.", "",
             "Statistics pool every frame-to-frame transition across all clips.", "",
             "| Configuration | Solution rate | Width jitter (p50 / p90) | Offset jitter | Heading jitter | Lane-count flips /100 | Empirical override |",
             "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        s = r["summary"]
        lines.append(
            f"| {name} | {s['solution_rate']*100:.1f}% "
            f"| {s['width_jitter_m']:.3f} / {s['width_jitter_p90_m']:.3f} m "
            f"| {s['offset_jitter_m']:.3f} m | {s['heading_jitter']:.4f} "
            f"| {s['count_flips_per_100']:.2f} | {s['empirical_frac']*100:.1f}% |")
    md = "\n".join(lines)
    (out_dir / "temporal.md").write_text(md)
    print("\n" + md)
    print(f"\nwritten to {out_dir}/temporal.{{md,json}}")


if __name__ == "__main__":
    main()
