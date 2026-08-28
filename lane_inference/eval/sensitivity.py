"""Sensitivity of lane inference to the metric calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config as cfg
from data.idd import IDDLite


def run(heights, data_root, split, limit, checkpoint):
    from core.pipeline import LaneInferencePipeline, PipelineOptions

    backend = None
    ckpt = Path(checkpoint) if checkpoint else cfg.default_checkpoint()
    if ckpt.is_file():
        from core.dl_drivable import SegmenterBackend
        backend = SegmenterBackend(str(ckpt))
        print(f"using learned segmentation ({ckpt.name})")
    else:
        print("no checkpoint found - using the classical backend")

    ds = IDDLite(data_root, split, input_size=(227, 320), augment=False)
    n = len(ds) if limit is None else min(limit, len(ds))
    frames = [ds.raw(i)[0] for i in range(n)]

    rows = []
    for H in heights:
        geom = cfg.GeometryConfig()
        geom.camera_height_m = float(H)


        opt = PipelineOptions(use_dl=backend is not None, use_cv=True)
        pipe = LaneInferencePipeline(backend=backend, options=opt, geom=geom)

        counts, widths, lane_w = [], [], []
        for f in frames:
            m = pipe.process(f).lane_model
            if m is None:
                continue
            counts.append(m.n_lanes)
            widths.append(m.carriageway_width_m)
            lane_w.append(m.lane_width_m)

        hist = {int(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))}
        rows.append({
            "camera_height_m": float(H),
            "solved": len(counts),
            "frames": n,
            "median_carriageway_m": float(np.median(widths)) if widths else float("nan"),
            "median_lane_width_m": float(np.median(lane_w)) if lane_w else float("nan"),
            "mean_lane_count": float(np.mean(counts)) if counts else float("nan"),
            "lane_count_hist": hist,
        })
        print(f"  H={H:.2f} m  solved {len(counts)}/{n}  "
              f"median carriageway {rows[-1]['median_carriageway_m']:.2f} m  "
              f"lanes {hist}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Calibration sensitivity sweep")
    p.add_argument("--heights", default="1.30,1.50,1.75,2.00,2.40")
    p.add_argument("--split", default="val")
    p.add_argument("--data-root", default=str(cfg.DATA_ROOT))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default=str(cfg.OUTPUT_ROOT / "results"))
    args = p.parse_args()

    heights = [float(h) for h in args.heights.split(",")]
    print(f"sweeping camera height over {heights}\n")
    rows = run(heights, args.data_root, args.split, args.limit, args.checkpoint)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity.json").write_text(json.dumps(rows, indent=2))

    lines = ["# Lane inference sensitivity to the metric calibration", "",
             "Camera height is the pipeline's only metric free parameter, and",
             "lateral scale is exactly proportional to it.", "",
             "| camera height | frames solved | median carriageway | median lane width | mean lane count | lane-count distribution |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['camera_height_m']:.2f} m | {r['solved']}/{r['frames']} "
            f"| {r['median_carriageway_m']:.2f} m | {r['median_lane_width_m']:.2f} m "
            f"| {r['mean_lane_count']:.2f} | "
            f"{', '.join(f'{k}:{v}' for k, v in sorted(r['lane_count_hist'].items()))} |")
    lines += ["", "## Reading this table", "",
              "**Coverage rises with assumed height.** A solution is rejected when the",
              "recovered carriageway falls below `min_lane_width_m` (2.5 m), and a larger",
              "assumed height scales every width up, so fewer frames are rejected. Coverage",
              "is therefore *not* independent of the calibration, and a high coverage number",
              "at a large assumed height is not evidence that the height is correct.", "",
              "**Median lane width is remarkably stable** across the whole sweep, because",
              "the lane count is an integer division: as the carriageway grows, the inferred",
              "count grows with it and the quotient stays near the IRC nominal. This is a",
              "genuine robustness property of partitioning rather than thresholding - the",
              "*structure* the pipeline reports degrades gracefully under scale error even",
              "though the absolute widths do not.", "",
              "**The lane count is what actually moves.** Between the extremes of the sweep",
              "the modal road shifts from single-lane to two-lane. Lane counts should",
              "therefore be read as conditional on the stated camera height, and a",
              "deployment that needs them exactly should fix the scale with",
              "`IPMTransform.calibrate_from_known_width()`."]
    md = "\n".join(lines)
    (out / "sensitivity.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
