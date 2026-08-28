"""Aggregate lane-inference statistics over a split."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import config as cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Descriptive lane-inference statistics")
    p.add_argument("--split", default="val")
    p.add_argument("--data-root", default=str(cfg.DATA_ROOT))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default=str(cfg.OUTPUT_ROOT / "results"))
    args = p.parse_args()

    from core.dl_drivable import SegmenterBackend
    from core.pipeline import LaneInferencePipeline, PipelineOptions
    from data.idd import IDDLite

    ds = IDDLite(args.data_root, args.split, input_size=(227, 320), augment=False)
    backend = SegmenterBackend(args.checkpoint)
    pipe = LaneInferencePipeline(backend=backend,
                                 options=PipelineOptions(use_dl=True, use_cv=True))
    print(f"model: {backend.checkpoint_path.name}, {len(ds)} frames\n")

    widths, lane_widths, counts, confs = [], [], [], []
    road_type, ego_assigned = Counter(), 0

    for i in range(len(ds)):
        r = pipe.process(ds.raw(i)[0])
        m = r.lane_model
        if m is None:
            continue
        widths.append(float(m.carriageway_width_m))
        lane_widths.append(float(m.lane_width_m))
        counts.append(int(m.n_lanes))
        confs.append(float(m.confidence))
        road_type[m.road_type] += 1
        if m.ego_lane >= 0:
            ego_assigned += 1

    res = {
        "solved": len(counts),
        "total": len(ds),
        "median_carriageway_m": float(np.median(widths)) if widths else None,
        "mean_lane_count": float(np.mean(counts)) if counts else None,
        "lane_hist": {str(k): v for k, v in sorted(Counter(counts).items())},
        "road_type": dict(road_type),
        "ego_assigned": ego_assigned,
        "mean_confidence": float(np.mean(confs)) if confs else None,
        "median_lane_width_m": float(np.median(lane_widths)) if lane_widths else None,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lane_inference_summary.json").write_text(json.dumps(res, indent=2))

    print(f"solution on {res['solved']} of {res['total']} frames")
    print(f"median carriageway   {res['median_carriageway_m']:.2f} m")
    print(f"median lane width    {res['median_lane_width_m']:.2f} m")
    print(f"mean lane count      {res['mean_lane_count']:.2f}")
    print(f"lane counts          {res['lane_hist']}")
    print(f"road type            {res['road_type']}")
    print(f"ego lane assigned    {res['ego_assigned']}")
    print(f"mean confidence      {res['mean_confidence']:.3f}")
    print(f"\nwritten to {out / 'lane_inference_summary.json'}")


if __name__ == "__main__":
    main()
