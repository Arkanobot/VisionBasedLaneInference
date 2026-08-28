"""Interactive demo for the lane-inference pipeline."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

import config as cfg
from core.pipeline import LaneInferencePipeline, PipelineOptions
from core.semantics import draw_road_users
from core.visualizer import (draw_bev_panel, draw_hud, draw_lane_overlay,
                             stack_panels)
from utils.image_utils import colorize_labels

STATE: dict = {}


def build_pipeline(backend_name: str, temporal: bool, camera_height: float):
    key = (backend_name, temporal, round(camera_height, 3))
    if key in STATE:
        return STATE[key]

    backend = None
    if backend_name != "cv":
        if "backend" not in STATE:
            from core.dl_drivable import SegmenterBackend
            STATE["backend"] = SegmenterBackend()
        backend = STATE["backend"]

    geom = cfg.GeometryConfig()
    geom.camera_height_m = camera_height
    pipe = LaneInferencePipeline(
        backend=backend,
        options=PipelineOptions(use_dl=backend is not None,
                                use_cv=True, use_temporal=temporal),
        geom=geom)
    STATE[key] = pipe
    return pipe


def summarise(result) -> str:
    m = result.lane_model
    if result.scene is not None and not result.scene:
        return ("### Not a road scene\n\n"
                f"**{result.scene.reason}.**\n\n"
                "Lane inference was not attempted. A segmentation network trained "
                "only on roads cannot abstain on its own — it labels a blank wall "
                "26% drivable — so the pipeline checks the structure of the scene "
                "before trusting it.\n\n"
                f"- semantic classes present: {result.scene.n_classes} (a road scene shows 5–7)\n"
                f"- drivable above the horizon: {result.scene.road_above_horizon*100:.0f}% "
                f"(should be 0%)\n"
                f"- drivable ahead of the camera: {result.scene.road_under_vehicle*100:.0f}%\n\n"
                f"Processed in {result.total_ms:.0f} ms.")
    if m is None:
        return ("### No lane solution\n\n"
                "The carriageway could not be measured. This happens on about "
                "15% of validation frames, most often where the road is too wide "
                "to fit within the frame, so neither visible boundary is a road "
                "edge. Traffic density does not explain it.\n\n"
                f"- camera pitch `{result.ipm.pitch_degrees:+.2f}°`\n"
                f"- processing `{result.total_ms:.0f} ms`")

    ego = f"{m.ego_lane + 1} of {m.n_lanes}" if m.ego_lane >= 0 else "not determined"
    conf_note = ("high — boundary error is typically below 0.6 m at this level"
                 if m.confidence >= 0.7 else
                 "moderate" if m.confidence >= 0.5 else
                 "low — treat the geometry as unreliable")
    lines = [
        f"### {m.road_type.replace('-', ' ').title()} road · **{m.n_lanes} lane"
        f"{'s' if m.n_lanes != 1 else ''}**", "",
        f"| | |", "|---|---|",
        f"| Carriageway | **{m.carriageway_width_m:.2f} m**"
        f"{' (lower bound)' if m.width_is_lower_bound else ''} |",
        f"| Lane width | {m.lane_width_m:.2f} m |",
        f"| Ego lane | {ego} |",
        f"| Confidence | **{m.confidence:.2f}** — {conf_note} |",
        f"| Boundary source | `{m.source}` |",
        f"| Camera pitch | {result.ipm.pitch_degrees:+.2f}° |",
    ]
    if result.traffic is not None and result.traffic.n_users:
        t = result.traffic
        lead = "clear" if not np.isfinite(t.lead_distance_m) else f"{t.lead_distance_m:.1f} m"
        lines += [f"| Road users | {t.n_users} |",
                  f"| Lead vehicle | {lead} |",
                  f"| Density | {t.density_per_100m:.0f} per 100 m |"]
    lines += ["", f"Processed in **{result.total_ms:.0f} ms** "
                  f"({result.fps:.0f} FPS) on `{cfg.resolve_device()}`.", "",
              "*Metric values assume a camera height of "
              f"{cfg.DEFAULT.geometry.camera_height_m} m and scale linearly with it.*"]
    return "\n".join(lines)


def compose(result, show_users: bool, max_range: float):
    img = result.image
    overlay = (draw_lane_overlay(img, result.lane_model, result.ipm, y_max=max_range)
               if result.lane_model else img.copy())
    if show_users and result.road_users:
        overlay = draw_road_users(overlay, result.road_users, result.wrong_side)
    overlay = draw_hud(overlay, result.lane_model, result.ipm,
                       {"fps": f"{result.fps:.0f}"}, top=6)

    panels, labels = [img, overlay], ["input", "lane inference"]
    if result.semantic is not None:
        panels.append(colorize_labels(result.semantic.labels.astype(np.uint8),
                                      cfg.CLASS_COLORS_BGR))
        labels.append("semantics")
    else:
        panels.append(cv2.cvtColor(result.road_mask, cv2.COLOR_GRAY2BGR))
        labels.append("road mask")
    if result.ipm is not None:
        panels.append(draw_bev_panel(result.ipm.warp_image(img),
                                     result.lane_model, result.ipm))
        labels.append("bird's-eye view")
    grid = stack_panels(panels, height=340, labels=labels)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)


def run_image(image, backend_name, camera_height, max_range, show_users):
    if image is None:
        return None, None, "Upload an image, or pick one of the samples below."
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if bgr.shape[1] > 1280:
        s = 1280 / bgr.shape[1]
        bgr = cv2.resize(bgr, (1280, int(bgr.shape[0] * s)))
    pipe = build_pipeline(backend_name, False, camera_height)
    pipe.reset()
    r = pipe.process(bgr)
    overlay, grid = compose(r, show_users, max_range)
    return overlay, grid, summarise(r)


def run_video(video, backend_name, camera_height, max_range, show_users,
              temporal, progress=gr.Progress()):
    if not video:
        return None, "Upload a clip, or pick one of the samples below."
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None, "Could not open that video."

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    pipe = build_pipeline(backend_name, temporal, camera_height)
    pipe.reset()

    out_path = str(Path(cfg.OUTPUT_ROOT) / "demo_output.mp4")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n, solved, lanes, widths, times = 0, 0, [], [], []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > 1280:
            s = 1280 / frame.shape[1]
            frame = cv2.resize(frame, (1280, int(frame.shape[0] * s)))
        r = pipe.process(frame)
        times.append(r.total_ms)
        if r.lane_model:
            solved += 1
            lanes.append(r.lane_model.n_lanes)
            widths.append(r.lane_model.carriageway_width_m)
        ov = (draw_lane_overlay(frame, r.lane_model, r.ipm, y_max=max_range)
              if r.lane_model else frame.copy())
        if show_users and r.road_users:
            ov = draw_road_users(ov, r.road_users, r.wrong_side)
        ov = draw_hud(ov, r.lane_model, r.ipm, {"fps": f"{r.fps:.0f}", "frame": n}, top=6)
        if writer is None:


            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"),
                                     fps_in, (ov.shape[1], ov.shape[0]))
            if not writer.isOpened():
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps_in, (ov.shape[1], ov.shape[0]))
        writer.write(ov)
        n += 1
        if total:
            progress(n / total, desc=f"frame {n}/{total}")
    cap.release()
    if writer:
        writer.release()
    if n == 0:
        return None, "No frames could be read from that video."

    flips = int(np.sum(np.asarray(lanes[1:]) != np.asarray(lanes[:-1]))) if len(lanes) > 1 else 0
    md = [f"### {n} frames processed", "", "| | |", "|---|---|",
          f"| Lane solution found | {solved}/{n} ({100*solved/n:.0f}%) |"]
    if lanes:
        md += [f"| Modal lane count | {int(np.bincount(lanes).argmax())} |",
               f"| Median carriageway | {np.median(widths):.2f} m |",
               f"| Lane-count changes | {flips} |"]
    md += [f"| Median latency | {np.median(times):.0f} ms ({1000/max(np.median(times),1e-6):.0f} FPS) |",
           f"| Temporal smoothing | {'on' if temporal else 'off'} |", "",
           "*With smoothing on, measured over 150 IDD Temporal sequences: width "
           "jitter falls 89% and lane-count changes 79%.*"]
    return out_path, "\n".join(md)


def run_camera(frame, backend_name, camera_height, max_range):
    if frame is None:
        return None
    bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
    if bgr.shape[1] > 960:
        s = 960 / bgr.shape[1]
        bgr = cv2.resize(bgr, (960, int(bgr.shape[0] * s)))
    pipe = build_pipeline(backend_name, True, camera_height)
    r = pipe.process(bgr)
    ov = (draw_lane_overlay(bgr, r.lane_model, r.ipm, y_max=max_range)
          if r.lane_model else bgr.copy())
    if r.road_users:
        ov = draw_road_users(ov, r.road_users, r.wrong_side)
    ov = draw_hud(ov, r.lane_model, r.ipm, {"fps": f"{r.fps:.0f}"}, top=6)
    return cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)


def _cell(img, caption, size=(420, 268)):
    """Fit an image into a fixed cell, letterboxed, with a caption bar."""
    W, H = size
    bar = 20
    canvas = np.zeros((H, W, 3), np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ih, iw = img.shape[:2]
    sc = min(W / iw, (H - bar) / ih)
    nw, nh = max(1, int(iw * sc)), max(1, int(ih * sc))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y0 = bar + (H - bar - nh) // 2
    x0 = (W - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    cv2.rectangle(canvas, (0, 0), (W, bar), (0, 0, 0), -1)
    cv2.putText(canvas, caption[:52], (5, bar - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), (45, 45, 45), 1)
    return canvas


def _heat(conf, label=None):
    """Render a 0-1 confidence map as an inferno heatmap."""
    m = np.clip(np.asarray(conf, dtype=np.float32), 0.0, 1.0)
    return cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def run_stages(image, backend_name, camera_height, max_range):
    """Every stage of the pipeline on one frame, in order."""
    if image is None:
        return None, "Upload an image, or pick a sample."
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if bgr.shape[1] > 1280:
        sc = 1280 / bgr.shape[1]
        bgr = cv2.resize(bgr, (1280, int(bgr.shape[0] * sc)))

    from core.asphalt_detector import detect_asphalt
    from core.geometry import road_edge_points, robust_line_fit
    from core.lane_tracker import extract_road_profile

    pipe = build_pipeline(backend_name, False, camera_height)
    pipe.reset()
    r = pipe.process(bgr)

    tiles, caps = [], []

    tiles.append(bgr); caps.append("1. input")

    _, cv_conf = detect_asphalt(bgr, return_confidence=True)
    tiles.append(_heat(cv_conf)); caps.append("2. classical road confidence")

    if r.semantic is not None:
        tiles.append(colorize_labels(r.semantic.labels.astype(np.uint8),
                                     cfg.CLASS_COLORS_BGR))
        caps.append("3. semantic segmentation")
        tiles.append(_heat(r.semantic.drivable_confidence))
        caps.append("4. learned drivable confidence")
        lm = r.semantic.lane_mask()
        if lm is not None:
            tiles.append(cv2.cvtColor(lm, cv2.COLOR_GRAY2BGR))
            caps.append("5. distilled lane-line head")

    tiles.append(cv2.cvtColor(r.road_mask, cv2.COLOR_GRAY2BGR))
    caps.append("6. road mask after cleanup")


    vis = bgr.copy()
    if r.vanishing_point is not None:
        rows, L, R, lv, rv = road_edge_points(r.road_mask)
        for y, x, ok in zip(rows, L, lv):
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0) if ok else (110, 110, 110), -1)
        for y, x, ok in zip(rows, R, rv):
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255) if ok else (110, 110, 110), -1)
        vp = r.vanishing_point
        cv2.line(vis, (0, int(vp.y)), (vis.shape[1] - 1, int(vp.y)), (255, 255, 0), 2)
        cv2.circle(vis, (int(vp.x), int(vp.y)), 7, (255, 255, 255), -1)
        cv2.circle(vis, (int(vp.x), int(vp.y)), 7, (0, 0, 0), 2)
    tiles.append(vis); caps.append("7. vanishing point + road edges")

    if r.ipm is not None:
        tiles.append(draw_bev_panel(r.ipm.warp_image(bgr), None, r.ipm))
        caps.append("8. rectified ground plane")

        prof_img = draw_bev_panel(cv2.cvtColor(r.ipm.warp_mask(r.road_mask),
                                               cv2.COLOR_GRAY2BGR), None, r.ipm)
        prof = extract_road_profile(r.ipm.warp_mask(r.road_mask), r.ipm)
        for Y, l, rr, ok in zip(prof.Y, prof.left, prof.right, prof.valid):
            if not (ok and np.isfinite(l) and np.isfinite(rr)):
                continue
            a = r.ipm.ground_to_bev([[l, Y]])[0]
            b = r.ipm.ground_to_bev([[rr, Y]])[0]
            cv2.line(prof_img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     (0, 255, 255), 1)
        tiles.append(prof_img); caps.append("9. width at each distance")

        if r.lane_model is not None:
            tiles.append(draw_bev_panel(r.ipm.warp_image(bgr), r.lane_model, r.ipm))
            caps.append("10. lanes partitioned in BEV")

    final = (draw_lane_overlay(bgr, r.lane_model, r.ipm, y_max=max_range)
             if r.lane_model else bgr.copy())
    if r.road_users:
        final = draw_road_users(final, r.road_users, r.wrong_side)
    tiles.append(final); caps.append("11. projected back to image")

    per_row = 3
    cells = [_cell(t, c) for t, c in zip(tiles, caps)]
    while len(cells) % per_row:
        cells.append(np.zeros_like(cells[0]))
    grid = np.concatenate([np.concatenate(cells[i:i + per_row], axis=1)
                           for i in range(0, len(cells), per_row)], axis=0)

    notes = [
        "### How the pipeline reaches its answer", "",
        "**2** The classical detector builds a colour model from the road directly "
        "ahead and gates it on local texture. On its own it scores 0.5795 IoU — "
        "below a fixed trapezoid that ignores the image entirely (0.6450). It is "
        "kept only as a fallback for when the learned branch collapses.", "",
        "**3–5** The network predicts seven classes plus a lane-line channel "
        "distilled from YOLOP. Drivable IoU 0.9349, lane-line IoU 0.5480.", "",
        "**7** Road edges are fitted per scanline; grey points are excluded because "
        "they sit on the frame border, where the observed boundary is the edge of "
        "the photograph rather than the edge of the road. The two fitted lines "
        "intersect at the vanishing point, which gives the camera pitch.", "",
        "**8** With pitch known, the ground plane is rectified. A metre is now a "
        "metre at every distance — in the original image a constant 7 m road spans "
        "385 px at 5 m and 55 px at 35 m.", "",
        "**9** Yellow lines are the measured carriageway at each forward distance. "
        "Only samples with two genuine road boundaries are used.", "",
        "**10** The carriageway is divided by the IRC design lane width. That "
        "integer division is what makes the output a lane *count* rather than an "
        "arbitrary subdivision.", "",
        f"Total {r.total_ms:.0f} ms — " +
        ", ".join(f"{k} {v:.0f} ms" for k, v in r.timings.items()) + ".",
    ]
    return cv2.cvtColor(grid, cv2.COLOR_BGR2RGB), "\n".join(notes)


def run_compare(image, camera_height, max_range):
    """Phase 3 pipeline against Phase 4, on the same frame."""
    if image is None:
        return None, None, "Upload an image, or pick a sample."
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if bgr.shape[1] > 1280:
        s = 1280 / bgr.shape[1]
        bgr = cv2.resize(bgr, (1280, int(bgr.shape[0] * s)))

    from core.visualizer import legacy_build_lane_overlay
    from eval.baselines import phase3_cv_only

    t0 = time.time()
    old_mask = phase3_cv_only(bgr)
    old = cv2.addWeighted(bgr, 0.6, legacy_build_lane_overlay(bgr, old_mask), 0.6, 0)
    old_ms = (time.time() - t0) * 1000

    pipe = build_pipeline("hybrid", False, camera_height)
    pipe.reset()
    r = pipe.process(bgr)
    new = (draw_lane_overlay(bgr, r.lane_model, r.ipm, y_max=max_range)
           if r.lane_model else bgr.copy())

    left = stack_panels([old, cv2.cvtColor(old_mask, cv2.COLOR_GRAY2BGR)], height=330,
                        labels=["Phase 3 lane output", "Phase 3 road mask"])
    right = stack_panels([new, cv2.cvtColor(r.road_mask, cv2.COLOR_GRAY2BGR)], height=330,
                         labels=["Phase 4 lane inference", "Phase 4 road mask"])

    m = r.lane_model
    md = [
        "### Same frame, both pipelines", "",
        "| | Phase 3 | Phase 4 |", "|---|---|---|",
        "| Road detection | fixed HSV threshold | network trained on IDD |",
        f"| Drivable IoU *(val set)* | 0.4131 | **0.9349** |",
        f"| Boundary F1 *(val set)* | 0.2093 | **0.8191** |",
        "| Lane method | bisect the widest row | metric partition in bird's-eye view |",
        f"| Lanes here | always 2 | **{m.n_lanes if m else 'no solution'}** |",
        f"| Carriageway | not measured | "
        f"**{f'{m.carriageway_width_m:.2f} m' if m else '—'}** |",
        f"| This frame | {old_ms:.0f} ms | {r.total_ms:.0f} ms |", "",
        "Phase 3's threshold accepts any desaturated mid-brightness pixel, which "
        "describes concrete, dust and overcast sky as readily as asphalt. On the "
        "validation set it scores **below a fixed trapezoid that ignores the image "
        "entirely** (0.6450).",
    ]
    return (cv2.cvtColor(left, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(right, cv2.COLOR_BGR2RGB), "\n".join(md))


def sample_images(k: int = 8):
    root = Path(cfg.DATA_ROOT) / "leftImg8bit" / "val"
    files = sorted(root.glob("*/*_image.jpg"))
    if not files:
        return []
    idx = np.linspace(0, len(files) - 1, min(k, len(files))).astype(int)
    return [str(files[i]) for i in idx]


def sample_videos(k: int = 6):
    root = Path(cfg.ROOT) / "datasets" / "idd_clips"
    files = sorted(root.glob("*.mp4"))
    if not files:
        return []
    idx = np.linspace(0, len(files) - 1, min(k, len(files))).astype(int)
    return [[str(files[i])] for i in idx]


def results_markdown() -> str:
    order = [("evaluation_val.md", "Drivable-area ablation"),
             ("lane_accuracy.md", "Carriageway accuracy and confidence"),
             ("temporal.md", "Temporal stability on real video"),
             ("multitask_ablation.md", "Distilled lane-line head"),
             ("scale_calibration.md", "Metric calibration"),
             ("geometry_ablation.md", "Vanishing point and rectification"),
             ("sensitivity.md", "Calibration sensitivity")]
    root = Path(cfg.OUTPUT_ROOT) / "results"
    parts = ["Measured results, regenerated by `python -m eval.evaluate` and the "
             "other harnesses in `eval/`. Nothing here is estimated.", ""]
    for name, title in order:
        p = root / name
        if p.is_file():
            parts += [f"\n---\n\n## {title}\n", p.read_text()]
    return "\n".join(parts) if len(parts) > 2 else "No result files found in outputs/results/."


CSS = """
.gradio-container {max-width: 1400px !important}
footer {display: none !important}

/* Phones. Gradio keeps columns side by side well below the width where that
   stops being readable, and the control row needs to wrap rather than squeeze. */
@media (max-width: 820px) {
  .gradio-container {padding: 6px !important}
  .gr-form, .gr-box {border-radius: 8px}
  /* stack every row */
  .gr-row, div[class*="row"] {flex-direction: column !important}
  .gr-column, div[class*="column"] {min-width: 100% !important; width: 100% !important}
  /* controls become full-width, tappable */
  .gr-button {min-height: 46px; font-size: 16px}
  input, select, textarea {font-size: 16px !important}  /* stops iOS zooming on focus */
  /* the four-panel strip is very wide; let it scroll instead of shrinking */
  .panel-strip img {max-width: none !important; height: auto}
  .panel-strip {overflow-x: auto}
  table {font-size: 13px}
}
"""


def build_ui(default_backend: str):
    with gr.Blocks(title="Lane Inference — Indian Roads", css=CSS,
                   theme=gr.themes.Soft(primary_hue="red", neutral_hue="slate")) as demo:
        gr.Markdown(
            "# Vision-Based Lane Inference for Indian Roads\n"
            "Inferring lane structure on roads **without lane markings**, by "
            "rectifying the drivable surface to a metric ground plane and "
            "partitioning it against the IRC design lane width.\n\n"
            "BITS Pilani · BSc Computer Science · Shreyas Bhat K · "
            "Harshwardhan Mukund Mohadikar")

        with gr.Row():
            backend = gr.Radio(["hybrid", "dl", "cv"], value=default_backend,
                               label="Backend",
                               info="hybrid = trained model with a classical fallback")
            height = gr.Slider(1.0, 2.5, cfg.DEFAULT.geometry.camera_height_m, step=0.05,
                               label="Camera height (m)",
                               info="the one metric calibration parameter; all "
                                    "distances scale with it")
            reach = gr.Slider(10, 30, 22, step=1, label="Overlay reach (m)")
            users = gr.Checkbox(True, label="Show road users")

        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_in = gr.Image(label="Road image", type="numpy", height=300)
                    img_btn = gr.Button("Analyse", variant="primary")
                    samples = sample_images()
                    if samples:
                        gr.Examples(samples, inputs=img_in, label="IDD validation frames")
                with gr.Column(scale=1):
                    img_txt = gr.Markdown()
            img_out = gr.Image(label="Lane inference", height=420)
            img_grid = gr.Image(label="Input · lanes · semantics · bird's-eye view")
            img_btn.click(run_image, [img_in, backend, height, reach, users],
                          [img_out, img_grid, img_txt])

        with gr.Tab("Video"):
            gr.Markdown("Temporal smoothing is what makes video output stable — "
                        "leave it on unless you want to see the difference.")
            with gr.Row():
                with gr.Column():
                    vid_in = gr.Video(label="Road video", height=300)
                    temporal = gr.Checkbox(True, label="Temporal smoothing (Kalman + lane-count vote)")
                    vid_btn = gr.Button("Process", variant="primary")
                    vids = sample_videos()
                    if vids:
                        gr.Examples(vids, inputs=vid_in, label="IDD Temporal sequences")
                with gr.Column():
                    vid_txt = gr.Markdown()
            vid_out = gr.Video(label="Annotated output", height=420)
            vid_btn.click(run_video,
                          [vid_in, backend, height, reach, users, temporal],
                          [vid_out, vid_txt])

        with gr.Tab("How it works"):
            gr.Markdown(
                "Every stage of the pipeline on one frame. Useful for seeing "
                "*why* an answer came out the way it did — and for seeing which "
                "stage is responsible when it is wrong.")
            with gr.Row():
                stg_in = gr.Image(label="Road image", type="numpy", height=280)
                stg_txt = gr.Markdown()
            stg_btn = gr.Button("Show every stage", variant="primary")
            stg_out = gr.Image(label="Pipeline stages", elem_classes="panel-strip")
            sm2 = sample_images()
            if sm2:
                gr.Examples(sm2, inputs=stg_in, label="IDD validation frames")
            stg_btn.click(run_stages, [stg_in, backend, height, reach],
                          [stg_out, stg_txt])

        with gr.Tab("Phase 3 vs Phase 4"):
            gr.Markdown(
                "The Phase 3 prototype and the current pipeline, run on the same "
                "frame. Phase 3 is reproduced exactly, including its `bitwise_or` "
                "fusion and its 120%-gain compositing.")
            with gr.Row():
                cmp_in = gr.Image(label="Road image", type="numpy", height=280)
                cmp_txt = gr.Markdown()
            cmp_btn = gr.Button("Compare", variant="primary")
            with gr.Row():
                cmp_old = gr.Image(label="Phase 3")
                cmp_new = gr.Image(label="Phase 4")
            sm = sample_images()
            if sm:
                gr.Examples(sm, inputs=cmp_in, label="IDD validation frames")
            cmp_btn.click(run_compare, [cmp_in, height, reach],
                          [cmp_old, cmp_new, cmp_txt])

        with gr.Tab("Camera"):
            gr.Markdown(
                "Live inference. Point the camera at a road, or at a screen "
                "playing dashcam footage.\n\n"
                "*The geometry assumes a forward-facing view of a road surface, so "
                "an indoor scene will produce no lane solution — that is correct "
                "behaviour, not a failure.*")
            with gr.Row():
                cam_in = gr.Image(sources=["webcam"], streaming=True,
                                  label="Camera", type="numpy", height=380)
                cam_out = gr.Image(label="Live inference", height=380)
            cam_in.stream(run_camera, [cam_in, backend, height, reach], cam_out,
                          stream_every=0.2, concurrency_limit=1)

        with gr.Tab("Results"):
            gr.Markdown(results_markdown())

    return demo


def local_address() -> str:
    """This machine's LAN address, so a phone on the same wifi can reach it."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.1.1", 1))
        addr = s.getsockname()[0]
        s.close()
        return addr
    except Exception:
        return "127.0.0.1"


def main():
    p = argparse.ArgumentParser(description="Lane inference demo")
    p.add_argument("--backend", default="hybrid", choices=["hybrid", "dl", "cv"])
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--lan", action="store_true",
                   help="serve on the local network so a phone can reach it")
    p.add_argument("--https", action="store_true",
                   help="serve over HTTPS using certs/. Browsers refuse camera "
                        "access over plain HTTP on anything but localhost, so this "
                        "is required for the camera tab to work on a phone.")
    p.add_argument("--share", action="store_true",
                   help="public gradio.live link. Convenient, but it proxies "
                        "through Gradio's servers; --lan --https keeps everything "
                        "on your own network.")
    args = p.parse_args()

    try:
        from core.dl_drivable import SegmenterBackend
        b = SegmenterBackend()
        print(f"model: {b.checkpoint_path.name} (val mIoU {b.trained_miou:.4f}) "
              f"on {b.device}")
    except FileNotFoundError:
        if args.backend != "cv":
            print("No checkpoint found - starting with the classical backend.")
            args.backend = "cv"

    kwargs = {"server_port": args.port, "share": args.share}
    scheme = "http"

    if args.lan or args.https:
        kwargs["server_name"] = "0.0.0.0"

    if args.https:
        certs = Path(__file__).parent / "certs"
        cert, key = certs / "cert.pem", certs / "key.pem"
        if not (cert.is_file() and key.is_file()):
            raise SystemExit(
                f"No certificate in {certs}. Generate one with:\n"
                f"  cd {certs} && openssl req -x509 -newkey rsa:2048 -nodes "
                f"-keyout key.pem -out cert.pem -days 365 -config san.cnf")
        kwargs["ssl_certfile"] = str(cert)
        kwargs["ssl_keyfile"] = str(key)
        kwargs["ssl_verify"] = False
        scheme = "https"

    if args.lan or args.https:
        addr = local_address()
        print("\n" + "=" * 58)
        print(f"  On this Mac : {scheme}://localhost:{args.port}")
        print(f"  On a phone  : {scheme}://{addr}:{args.port}")
        if args.https:
            print("\n  The certificate is self-signed, so the phone will warn once.")
            print('  Tap "Advanced" then "Proceed" - it is your own machine.')
            print("  The camera tab needs this; browsers block cameras on plain HTTP.")
        else:
            print("\n  Uploads work. The camera tab will NOT - browsers block")
            print("  camera access over plain HTTP. Add --https for the camera.")
        print("  Phone and Mac must be on the same wifi.")
        print("=" * 58 + "\n")

    build_ui(args.backend).launch(**kwargs)


if __name__ == "__main__":
    main()
