"""Vision-based lane inference for unstructured and structured Indian roads."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import config as cfg
from core.pipeline import LaneInferencePipeline, PipelineOptions
from core.semantics import draw_road_users
from core.visualizer import (draw_bev_panel, draw_hud, draw_lane_overlay,
                             stack_panels)
from utils.image_utils import colorize_labels


def make_pipeline(args) -> LaneInferencePipeline:
    backend = None
    if args.backend in ("dl", "hybrid"):
        from core.dl_drivable import SegmenterBackend
        try:
            backend = SegmenterBackend(args.checkpoint)
            print(f"model: {backend.checkpoint_path.name} "
                  f"(val mIoU {backend.trained_miou:.4f}) on {backend.device}")
        except FileNotFoundError as e:
            print(f"{e}\nFalling back to the classical backend.\n")
            args.backend = "cv"


    opt = PipelineOptions(
        use_dl=backend is not None,
        use_cv=args.backend in ("cv", "hybrid"),
        use_fusion=args.blend and args.backend == "hybrid",
        use_gated_fallback=(not args.blend) and args.backend == "hybrid",
        use_temporal=args.temporal,
        use_markings=not args.no_markings,
        use_trajectory_lanes=not args.no_trajectory_lanes,
        detect_wrong_side=args.wrong_side,
    )
    geom = cfg.GeometryConfig()
    geom.camera_height_m = args.camera_height
    return LaneInferencePipeline(backend=backend, options=opt, geom=geom)


def render(result, args, extra: dict | None = None) -> np.ndarray:
    """Build the display frame: overlay alone, or the full diagnostic panel."""
    image = result.image
    if result.lane_model is not None and result.ipm is not None:
        overlay = draw_lane_overlay(image, result.lane_model, result.ipm,
                                    y_max=args.max_range)
    else:
        overlay = image.copy()
    if result.road_users and not args.no_users:
        overlay = draw_road_users(overlay, result.road_users, result.wrong_side)

    info = {"fps": f"{result.fps:.0f}"}
    if result.traffic is not None and result.traffic.n_users:
        t = result.traffic
        lead = "clear" if not np.isfinite(t.lead_distance_m) else f"{t.lead_distance_m:.0f}m"
        info["traffic"] = f"{t.n_users} users, lead {lead}"
    if result.lane_source == "empirical":
        info["lanes from"] = "observed traffic"
    if result.wrong_side:
        info["WRONG SIDE"] = f"{len(result.wrong_side)} vehicle(s)"
    info.update(extra or {})
    overlay = draw_hud(overlay, result.lane_model, result.ipm, info)\
        if result.ipm is not None else overlay

    if not args.panel:
        return overlay

    panels = [image, overlay]
    labels = ["input", "lane inference"]
    if result.semantic is not None:
        panels.append(colorize_labels(result.semantic.labels.astype(np.uint8),
                                      cfg.CLASS_COLORS_BGR))
        labels.append("semantics")
    else:
        panels.append(cv2.cvtColor(result.road_mask, cv2.COLOR_GRAY2BGR))
        labels.append("road mask")
    if result.ipm is not None:
        panels.append(draw_bev_panel(result.ipm.warp_image(image),
                                     result.lane_model, result.ipm))
        labels.append("rectified view")
    return stack_panels(panels, height=args.panel_height, labels=labels)


def describe(result) -> str:
    m = result.lane_model
    if m is None:
        return "  no lane solution"
    ego = f"{m.ego_lane + 1} of {m.n_lanes}" if m.ego_lane >= 0 else "unknown"
    return (f"  road type      : {m.road_type}\n"
            f"  lanes          : {m.n_lanes}\n"
            f"  carriageway    : {m.carriageway_width_m:.2f} m\n"
            f"  lane width     : {m.lane_width_m:.2f} m\n"
            f"  ego lane       : {ego}\n"
            f"  confidence     : {m.confidence:.2f}  [{m.source}]\n"
            f"  lane source    : {result.lane_source}\n"
            f"  camera pitch   : {result.ipm.pitch_degrees:+.2f} deg")


def cmd_image(args) -> int:
    image = cv2.imread(args.path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"error: cannot read image '{args.path}'", file=sys.stderr)
        return 1
    if args.resize:
        w = args.resize
        image = cv2.resize(image, (w, int(image.shape[0] * w / image.shape[1])))

    pipe = make_pipeline(args)
    result = pipe.process(image)

    print(f"\n{Path(args.path).name}  {image.shape[1]}x{image.shape[0]}")
    print(describe(result))
    print("  stage timings  : " + ", ".join(f"{k} {v:.1f}ms"
                                            for k, v in result.timings.items()))
    print(f"  total          : {result.total_ms:.1f} ms  ({result.fps:.0f} FPS)")

    frame = render(result, args)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.save, frame)
        print(f"\nsaved {args.save}")
    if not args.no_show:
        cv2.imshow("lane inference", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def cmd_video(args) -> int:
    cap = cv2.VideoCapture(args.path if not args.path.isdigit() else int(args.path))
    if not cap.isOpened():
        print(f"error: cannot open '{args.path}'", file=sys.stderr)
        return 1

    args.temporal = True
    pipe = make_pipeline(args)
    writer = None
    n, t_start, fps_hist = 0, time.time(), []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.resize:
                w = args.resize
                frame = cv2.resize(frame, (w, int(frame.shape[0] * w / frame.shape[1])))

            result = pipe.process(frame)
            fps_hist.append(result.fps)
            out = render(result, args, {"frame": n})

            if args.save and writer is None:
                Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(
                    args.save, fourcc, cap.get(cv2.CAP_PROP_FPS) or 25.0,
                    (out.shape[1], out.shape[0]))
                if not writer.isOpened():
                    writer = cv2.VideoWriter(
                        args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                        cap.get(cv2.CAP_PROP_FPS) or 25.0,
                        (out.shape[1], out.shape[0]))
            if writer is not None:
                writer.write(out)
            if not args.no_show:
                cv2.imshow("lane inference", out)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            n += 1
            if n % 30 == 0:
                print(f"  frame {n}  {np.mean(fps_hist[-30:]):.1f} FPS", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    wall = time.time() - t_start
    print(f"\n{n} frames in {wall:.1f}s  "
          f"({n/max(wall,1e-6):.1f} FPS end to end, "
          f"{np.median(fps_hist) if fps_hist else float('nan'):.1f} FPS pipeline median)")
    if args.save:
        print(f"saved {args.save}")
    return 0


def cmd_dataset(args) -> int:
    from data.idd import IDDLite
    ds = IDDLite(cfg.DATA_ROOT, args.split, input_size=(227, 320), augment=False)
    pipe = make_pipeline(args)
    idxs = (np.linspace(0, len(ds) - 1, args.count).astype(int)
            if args.indices is None else
            [int(i) for i in args.indices.split(",")])

    rows = []
    for i in idxs:
        image, _ = ds.raw(int(i))
        result = pipe.process(image)
        rows.append(render(result, args))
        m = result.lane_model
        print(f"  #{i:<4} " + (f"{m.road_type:<13} {m.n_lanes} lanes  "
                               f"{m.carriageway_width_m:5.2f} m  conf {m.confidence:.2f}"
                               if m else "no solution"))
    width = max(r.shape[1] for r in rows)
    grid = np.concatenate([np.pad(r, ((0, 6), (0, width - r.shape[1]), (0, 0)))
                           for r in rows], axis=0)
    out = args.save or str(cfg.OUTPUT_ROOT / f"dataset_{args.split}.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, grid)
    print(f"\nsaved {out}  ({grid.shape[1]}x{grid.shape[0]})")
    return 0


def cmd_benchmark(args) -> int:
    from data.idd import IDDLite
    ds = IDDLite(cfg.DATA_ROOT, "val", input_size=(227, 320), augment=False)
    frames = [ds.raw(i)[0] for i in range(min(args.count, len(ds)))]

    print(f"benchmarking on {len(frames)} frames, device "
          f"{cfg.resolve_device()}\n")
    for backend in (["cv", "hybrid"] if args.backend == "all" else [args.backend]):
        args.backend = backend
        try:
            pipe = make_pipeline(args)
        except Exception as e:
            print(f"{backend}: unavailable ({e})")
            continue
        for f in frames[:5]:
            pipe.process(f)
        stage_totals, totals = {}, []
        for f in frames:
            r = pipe.process(f)
            totals.append(r.total_ms)
            for k, v in r.timings.items():
                stage_totals[k] = stage_totals.get(k, 0.0) + v
        n = len(frames)
        print(f"--- backend: {backend} ---")
        for k, v in stage_totals.items():
            print(f"  {k:<16}{v/n:7.2f} ms")
        print(f"  {'TOTAL':<16}{np.mean(totals):7.2f} ms   "
              f"median {np.median(totals):.2f} ms   "
              f"{1000/np.median(totals):.1f} FPS\n")
    return 0


def _common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--backend", default="hybrid", choices=["cv", "dl", "hybrid", "all"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--camera-height", type=float,
                   default=cfg.DEFAULT.geometry.camera_height_m,
                   help="metric calibration; see outputs/results/scale_calibration.md")
    p.add_argument("--max-range", type=float, default=22.0,
                   help="furthest distance (m) drawn in the overlay")
    p.add_argument("--panel", action="store_true", help="show the diagnostic panel")
    p.add_argument("--panel-height", type=int, default=360)
    p.add_argument("--temporal", action="store_true")
    p.add_argument("--blend", action="store_true",
                   help="blend CV and DL confidence instead of gated fallback "
                        "(ablation only: measured to reduce IoU)")
    p.add_argument("--no-markings", action="store_true")
    p.add_argument("--no-users", action="store_true", help="hide road-user boxes")
    p.add_argument("--wrong-side", action="store_true",
                   help="flag vehicles travelling against the flow (video, experimental)")
    p.add_argument("--no-trajectory-lanes", action="store_true",
                   help="disable empirical lane inference from observed traffic")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--save", default=None)
    p.add_argument("--resize", type=int, default=None, help="resize width")
    return p


def build_parser() -> argparse.ArgumentParser:


    common = _common_args(argparse.ArgumentParser(add_help=False))

    p = argparse.ArgumentParser(
        parents=[common],
        description="Vision-based lane inference for Indian roads",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)

    sub = p.add_subparsers(dest="command")
    sp = sub.add_parser("image", parents=[common])
    sp.add_argument("path"); sp.set_defaults(func=cmd_image)
    sp = sub.add_parser("video", parents=[common])
    sp.add_argument("path"); sp.set_defaults(func=cmd_video)
    sp = sub.add_parser("camera", parents=[common])
    sp.add_argument("path", nargs="?", default="0"); sp.set_defaults(func=cmd_video)
    sp = sub.add_parser("dataset", parents=[common])
    sp.add_argument("--split", default="val"); sp.add_argument("--count", type=int, default=8)
    sp.add_argument("--indices", default=None); sp.set_defaults(func=cmd_dataset)
    sp = sub.add_parser("benchmark", parents=[common])
    sp.add_argument("--count", type=int, default=50); sp.set_defaults(func=cmd_benchmark)
    return p


def main() -> int:
    argv = sys.argv[1:]

    if argv and not argv[0].startswith("-") and\
       argv[0] not in {"image", "video", "camera", "dataset", "benchmark"} and\
       Path(argv[0]).is_file():
        argv = ["image"] + argv

    args = build_parser().parse_args(argv)
    if not hasattr(args, "func"):
        build_parser().print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
