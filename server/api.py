"""HTTP API for the lane-inference pipeline."""
from __future__ import annotations

import base64
import io
import json
import math
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lane_inference"))

import config as cfg                                            # noqa: E402
from core.pipeline import LaneInferencePipeline, PipelineOptions  # noqa: E402
from core.semantics import draw_road_users                       # noqa: E402
from core.visualizer import (draw_bev_panel, draw_lane_overlay,   # noqa: E402
                             legacy_build_lane_overlay)
from utils.image_utils import colorize_labels, overlay_mask      # noqa: E402

def _finite(obj):
    """Replace non-finite floats with None, recursively."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    """JSON response that cannot be killed by a non-finite float."""

    def render(self, content) -> bytes:
        return super().render(_finite(content))


app = FastAPI(title="Lane Inference API", version="1.0",
              default_response_class=SafeJSONResponse)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

STATE: dict = {}
SESSIONS: dict = {}
MEDIA = ROOT / "outputs" / "api"
MEDIA.mkdir(parents=True, exist_ok=True)


def backend():
    if "backend" not in STATE:
        try:
            from core.dl_drivable import SegmenterBackend
            STATE["backend"] = SegmenterBackend()
        except FileNotFoundError:
            STATE["backend"] = None
    return STATE["backend"]


def pipeline(camera_height: float = 1.75, temporal: bool = False, key=None):
    if key is not None:
        if key not in SESSIONS:
            geom = cfg.GeometryConfig(); geom.camera_height_m = camera_height
            SESSIONS[key] = LaneInferencePipeline(
                backend=backend(),
                options=PipelineOptions(use_dl=backend() is not None,
                                        use_cv=True, use_temporal=True),
                geom=geom)
        return SESSIONS[key]

    k = ("single", round(camera_height, 3), temporal)
    if k not in STATE:
        geom = cfg.GeometryConfig(); geom.camera_height_m = camera_height
        STATE[k] = LaneInferencePipeline(
            backend=backend(),
            options=PipelineOptions(use_dl=backend() is not None,
                                    use_cv=True, use_temporal=temporal),
            geom=geom)
    return STATE[k]


def to_b64(bgr: np.ndarray, quality: int = 86) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def read_upload(data: bytes, max_width: int = 1280) -> np.ndarray:


    if not data:
        raise HTTPException(400, "Empty upload")
    arr = np.frombuffer(data, np.uint8)
    try:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except cv2.error:
        img = None
    if img is None:
        raise HTTPException(400, "Could not decode that image")
    if img.shape[1] > max_width:
        s = max_width / img.shape[1]
        img = cv2.resize(img, (max_width, int(img.shape[0] * s)),
                         interpolation=cv2.INTER_AREA)
    return img


def clamp_range(v: float, lo: float = 5.0, hi: float = 60.0) -> float:
    """Keep the requested overlay reach finite and sane."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 22.0
    if not math.isfinite(v):
        return 22.0
    return float(min(max(v, lo), hi))


def result_payload(r, max_range: float) -> dict:
    """Everything the frontend needs about one processed frame."""
    max_range = clamp_range(max_range)
    m = r.lane_model
    scene_ok = r.scene is None or bool(r.scene)

    payload = {
        "ok": m is not None,
        "scene": {
            "isRoad": scene_ok,
            "reason": None if r.scene is None else r.scene.reason,
            "classes": None if r.scene is None else r.scene.n_classes,
            "roadAboveHorizon": None if r.scene is None else r.scene.road_above_horizon,
        },
        "timings": {k: round(v, 2) for k, v in r.timings.items()},
        "totalMs": round(r.total_ms, 1),
        "fps": round(r.fps, 1),
        "device": cfg.resolve_device(),
        "pitchDeg": None if r.ipm is None else round(r.ipm.pitch_degrees, 2),
        "cameraHeightM": cfg.DEFAULT.geometry.camera_height_m,
    }

    if m is not None:
        payload["lane"] = {
            "count": m.n_lanes,
            "widthM": round(m.carriageway_width_m, 2),
            "laneWidthM": round(m.lane_width_m, 2),
            "egoLane": m.ego_lane + 1 if m.ego_lane >= 0 else None,
            "roadType": m.road_type,
            "confidence": round(m.confidence, 3),
            "source": m.source,
            "widthIsLowerBound": bool(m.width_is_lower_bound),
        }

    if r.traffic is not None:
        t = r.traffic
        payload["traffic"] = {
            "users": t.n_users,
            "perLane": {str(k): v for k, v in t.per_lane.items()},
            "densityPer100m": round(t.density_per_100m, 1),
            "leadDistanceM": None if not np.isfinite(t.lead_distance_m)
            else round(t.lead_distance_m, 1),
        }

    payload["roadUsers"] = [
        {"x": round(u.x_m, 2), "y": round(u.y_m, 2), "widthM": round(u.width_m, 2),
         "lane": u.lane + 1 if u.lane >= 0 else None,
         "cls": "vehicle" if u.class_id == cfg.VEHICLE_ID else "person",
         "bbox": list(u.bbox)}
        for u in r.road_users
    ]
    return payload


def render_views(r, max_range: float) -> dict:
    max_range = clamp_range(max_range)
    img = r.image
    overlay = (draw_lane_overlay(img, r.lane_model, r.ipm, y_max=max_range)
               if r.lane_model else img.copy())
    if r.road_users:
        overlay = draw_road_users(overlay, r.road_users, r.wrong_side)
    views = {"input": to_b64(img), "lanes": to_b64(overlay)}
    if r.semantic is not None:
        views["semantics"] = to_b64(colorize_labels(
            r.semantic.labels.astype(np.uint8), cfg.CLASS_COLORS_BGR))
    views["mask"] = to_b64(overlay_mask(img, r.road_mask, (60, 220, 60), 0.5))
    if r.ipm is not None:
        views["bev"] = to_b64(draw_bev_panel(r.ipm.warp_image(img),
                                             r.lane_model, r.ipm))
    return views


@app.get("/api/health")
def health():
    b = backend()
    return {
        "status": "ok",
        "device": cfg.resolve_device(),
        "model": None if b is None else {
            "checkpoint": b.checkpoint_path.name,
            "run": b.checkpoint_path.parent.name,
            "valMIoU": round(b.trained_miou, 4),
            "laneHead": bool(getattr(b, "has_lane_head", False)),
            "inputSize": list(b.input_size),
        },
        "cameraHeightM": cfg.DEFAULT.geometry.camera_height_m,
        "classes": cfg.CLASS_NAMES,
    }


@app.post("/api/infer")
async def infer(file: UploadFile = File(...),
                cameraHeight: float = Form(1.75),
                maxRange: float = Form(22.0)):
    img = read_upload(await file.read())
    p = pipeline(cameraHeight)
    p.reset()
    t0 = time.perf_counter()
    r = p.process(img)
    payload = result_payload(r, maxRange)
    payload["views"] = render_views(r, maxRange)
    payload["wallMs"] = round((time.perf_counter() - t0) * 1000, 1)
    return payload


@app.post("/api/live")
async def live(file: UploadFile = File(...),
               session: str = Form("default"),
               cameraHeight: float = Form(1.75),
               maxRange: float = Form(22.0),
               width: int = Form(640)):
    """One frame of a live camera stream."""
    img = read_upload(await file.read(), max_width=width)
    p = pipeline(cameraHeight, temporal=True, key=session)
    r = p.process(img)

    ov = (draw_lane_overlay(img, r.lane_model, r.ipm, y_max=clamp_range(maxRange))
          if r.lane_model else img.copy())
    if r.road_users:
        ov = draw_road_users(ov, r.road_users, r.wrong_side)

    m = r.lane_model
    return {
        "frame": to_b64(ov, 72),
        "ok": m is not None,
        "isRoad": r.scene is None or bool(r.scene),
        "reason": None if (r.scene is None or bool(r.scene)) else r.scene.reason,
        "lanes": m.n_lanes if m else None,
        "widthM": round(m.carriageway_width_m, 1) if m else None,
        "laneWidthM": round(m.lane_width_m, 2) if m else None,
        "egoLane": (m.ego_lane + 1) if (m and m.ego_lane >= 0) else None,
        "confidence": round(m.confidence, 2) if m else None,
        "roadType": m.road_type if m else None,
        "users": len(r.road_users),
        "leadM": (None if r.traffic is None or not np.isfinite(r.traffic.lead_distance_m)
                  else round(r.traffic.lead_distance_m, 1)),
        "nearest": ([round(u.y_m, 1) for u in sorted(r.road_users, key=lambda u: u.y_m)[:3]]),
        "ms": round(r.total_ms, 1),
        "pitchDeg": None if r.ipm is None else round(r.ipm.pitch_degrees, 1),
    }


@app.post("/api/live/reset")
async def live_reset(session: str = Form("default")):
    SESSIONS.pop(session, None)
    return {"reset": session}


@app.post("/api/stages")
async def stages(file: UploadFile = File(...),
                 cameraHeight: float = Form(1.75),
                 maxRange: float = Form(22.0)):
    """Every pipeline stage, for the walkthrough view."""
    from core.asphalt_detector import detect_asphalt
    from core.geometry import road_edge_points
    from core.lane_tracker import extract_road_profile

    img = read_upload(await file.read())
    p = pipeline(cameraHeight)
    p.reset()
    r = p.process(img)

    def heat(c):
        return cv2.applyColorMap(
            (np.clip(c, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    out = []
    out.append({"key": "input", "title": "Input frame", "image": to_b64(img),
                "note": "A single forward-facing photograph. Nothing else is assumed."})

    _, cv_conf = detect_asphalt(img, return_confidence=True)
    out.append({"key": "classical", "title": "Classical road confidence",
                "image": to_b64(heat(cv_conf)),
                "note": "A colour model seeded from the road directly ahead, gated on "
                        "local texture. Alone it scores 0.5795 IoU — below a fixed "
                        "trapezoid that ignores the image entirely (0.6450). Kept only "
                        "as a fallback for when the learned branch collapses."})

    if r.semantic is not None:
        out.append({"key": "semantics", "title": "Semantic segmentation",
                    "image": to_b64(colorize_labels(r.semantic.labels.astype(np.uint8),
                                                    cfg.CLASS_COLORS_BGR)),
                    "note": "Seven classes from a network trained on the Indian Driving "
                            "Dataset. Drivable IoU 0.9349."})
        out.append({"key": "drivable", "title": "Drivable confidence",
                    "image": to_b64(heat(r.semantic.drivable_confidence)),
                    "note": "Per-pixel probability that the surface is drivable."})
        lm = r.semantic.lane_mask()
        if lm is not None:
            out.append({"key": "lanelines", "title": "Lane-line head",
                        "image": to_b64(cv2.cvtColor(lm, cv2.COLOR_GRAY2BGR)),
                        "note": "Distilled from YOLOP's lane-line head, which the "
                                "Phase 3 pipeline never read. Runs in the same forward "
                                "pass instead of YOLOP's separate 150 ms."})

    out.append({"key": "mask", "title": "Road mask",
                "image": to_b64(overlay_mask(img, r.road_mask, (60, 220, 60), 0.5)),
                "note": "After cleanup and a bottom-connectivity constraint — the "
                        "vehicle is standing on the road, so the region must reach the "
                        "bottom of the frame."})

    vis = img.copy()
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
    out.append({"key": "vp", "title": "Vanishing point", "image": to_b64(vis),
                "note": "Road edges fitted per scanline. Grey points are excluded "
                        "because they sit on the frame border, where the observed "
                        "boundary is the edge of the photograph, not of the road. The "
                        "two fitted lines intersect at the vanishing point, which gives "
                        "the camera pitch."})

    if r.ipm is not None:
        out.append({"key": "bev", "title": "Rectified ground plane",
                    "image": to_b64(draw_bev_panel(r.ipm.warp_image(img), None, r.ipm)),
                    "note": "With pitch known, the ground plane is rectified. A metre "
                            "is a metre at every distance — in the original image a "
                            "constant 7 m road spans 385 px at 5 m and 55 px at 35 m."})

        prof_img = draw_bev_panel(cv2.cvtColor(r.ipm.warp_mask(r.road_mask),
                                               cv2.COLOR_GRAY2BGR), None, r.ipm)
        prof = extract_road_profile(r.ipm.warp_mask(r.road_mask), r.ipm)
        for Y, l, rr, ok in zip(prof.Y, prof.left, prof.right, prof.valid):
            if ok and np.isfinite(l) and np.isfinite(rr):
                a = r.ipm.ground_to_bev([[l, Y]])[0]
                b = r.ipm.ground_to_bev([[rr, Y]])[0]
                cv2.line(prof_img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                         (0, 255, 255), 1)
        out.append({"key": "profile", "title": "Carriageway width",
                    "image": to_b64(prof_img),
                    "note": "Measured at each forward distance. Only samples with two "
                            "genuine road boundaries count; the rest are discarded."})

        if r.lane_model is not None:
            out.append({"key": "partition", "title": "Lane partition",
                        "image": to_b64(draw_bev_panel(r.ipm.warp_image(img),
                                                       r.lane_model, r.ipm)),
                        "note": "The carriageway divided by the IRC design lane width. "
                                "That integer division is what makes the output a lane "
                                "count rather than an arbitrary subdivision."})

    final = (draw_lane_overlay(img, r.lane_model, r.ipm, y_max=clamp_range(maxRange))
             if r.lane_model else img.copy())
    if r.road_users:
        final = draw_road_users(final, r.road_users, r.wrong_side)
    out.append({"key": "final", "title": "Projected back", "image": to_b64(final),
                "note": "Lane boundaries mapped from the ground plane back into the "
                        "camera image."})

    return {"stages": out, "result": result_payload(r, maxRange)}


@app.post("/api/compare")
async def compare(file: UploadFile = File(...),
                  cameraHeight: float = Form(1.75),
                  maxRange: float = Form(22.0)):
    from eval.baselines import phase3_cv_only

    img = read_upload(await file.read())
    t0 = time.perf_counter()
    old_mask = phase3_cv_only(img)
    old = cv2.addWeighted(img, 0.6, legacy_build_lane_overlay(img, old_mask), 0.6, 0)
    old_ms = (time.perf_counter() - t0) * 1000

    p = pipeline(cameraHeight); p.reset()
    r = p.process(img)
    new = (draw_lane_overlay(img, r.lane_model, r.ipm, y_max=clamp_range(maxRange))
           if r.lane_model else img.copy())

    return {
        "phase3": {"lanes": to_b64(old), "mask": to_b64(old_mask), "ms": round(old_ms, 1),
                   "drivableIoU": 0.4131, "boundaryF1": 0.2093, "laneMethod":
                   "widest row, bisected"},
        "phase4": {"lanes": to_b64(new), "mask": to_b64(r.road_mask),
                   "ms": round(r.total_ms, 1), "drivableIoU": 0.9349,
                   "boundaryF1": 0.8191, "laneMethod":
                   "metric partition in bird's-eye view"},
        "result": result_payload(r, maxRange),
    }


@app.post("/api/video")
async def video(file: UploadFile = File(...),
                cameraHeight: float = Form(1.75),
                maxRange: float = Form(22.0),
                temporal: bool = Form(True)):
    vid_id = uuid.uuid4().hex[:12]
    src = MEDIA / f"in_{vid_id}.mp4"
    src.write_bytes(await file.read())

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        src.unlink(missing_ok=True)
        raise HTTPException(400, "Could not open that video")

    geom = cfg.GeometryConfig(); geom.camera_height_m = cameraHeight
    p = LaneInferencePipeline(
        backend=backend(),
        options=PipelineOptions(use_dl=backend() is not None, use_cv=True,
                                use_temporal=temporal),
        geom=geom)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 10.0
    out_path = MEDIA / f"out_{vid_id}.mp4"
    writer = None
    n = solved = 0
    lanes, widths, times, series = [], [], [], []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > 1280:
            s = 1280 / frame.shape[1]
            frame = cv2.resize(frame, (1280, int(frame.shape[0] * s)))
        r = p.process(frame)
        times.append(r.total_ms)
        m = r.lane_model
        if m:
            solved += 1
            lanes.append(m.n_lanes)
            widths.append(m.carriageway_width_m)
        series.append({"f": n,
                       "width": round(m.carriageway_width_m, 2) if m else None,
                       "lanes": m.n_lanes if m else None,
                       "conf": round(m.confidence, 3) if m else None})
        ov = (draw_lane_overlay(frame, m, r.ipm, y_max=clamp_range(maxRange)) if m else frame.copy())
        if r.road_users:
            ov = draw_road_users(ov, r.road_users, r.wrong_side)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps_in,
                                     (ov.shape[1], ov.shape[0]))
            if not writer.isOpened():
                writer = cv2.VideoWriter(str(out_path),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps_in, (ov.shape[1], ov.shape[0]))
        writer.write(ov)
        n += 1
    cap.release()
    if writer:
        writer.release()
    src.unlink(missing_ok=True)
    if n == 0:
        raise HTTPException(400, "No frames could be read")

    flips = int(np.sum(np.asarray(lanes[1:]) != np.asarray(lanes[:-1]))) if len(lanes) > 1 else 0
    return {
        "videoUrl": f"/api/media/{out_path.name}",
        "frames": n,
        "solved": solved,
        "solutionRate": round(solved / n, 3),
        "modalLanes": int(np.bincount(lanes).argmax()) if lanes else None,
        "medianWidthM": round(float(np.median(widths)), 2) if widths else None,
        "laneCountChanges": flips,
        "medianMs": round(float(np.median(times)), 1),
        "temporal": temporal,
        "series": series,
    }


@app.get("/api/media/{name}")
def media(name: str):
    f = MEDIA / name
    if not f.is_file() or ".." in name or "/" in name:
        raise HTTPException(404, "not found")
    return FileResponse(f, media_type="video/mp4")


@app.get("/api/samples")
def samples(kind: str = "image", n: int = 8):
    if kind == "video":
        files = sorted((ROOT / "datasets" / "idd_clips").glob("*.mp4"))
    else:
        files = sorted((cfg.DATA_ROOT / "leftImg8bit" / "val").glob("*/*_image.jpg"))
    if not files:
        return {"samples": []}
    idx = np.linspace(0, len(files) - 1, min(n, len(files))).astype(int)
    out = []
    for i in idx:
        f = files[int(i)]
        item = {"id": f.stem, "name": f.name, "url": f"/api/sample/{kind}/{f.stem}"}
        if kind == "image":
            im = cv2.imread(str(f))
            if im is not None:
                item["thumb"] = to_b64(cv2.resize(im, (192, 108)), 70)
        out.append(item)
    return {"samples": out}


@app.get("/api/sample/{kind}/{stem}")
def sample_file(kind: str, stem: str):
    root = (ROOT / "datasets" / "idd_clips") if kind == "video" else\
        (cfg.DATA_ROOT / "leftImg8bit" / "val")
    pattern = "*.mp4" if kind == "video" else "*/*_image.jpg"
    for f in root.glob(pattern):
        if f.stem == stem:
            return FileResponse(f)
    raise HTTPException(404, "not found")


@app.get("/api/results")
def results():
    """Every measured result, for the dashboard."""
    rd = ROOT / "outputs" / "results"
    out = {"tables": {}, "docs": {}}
    for name in ["evaluation_val.json", "sensitivity.json", "lane_geometry.json",
                 "temporal.json", "failures.json", "empirical_sweep.json",
                 "idd20k_history.json", "lane_inference_summary.json"]:
        f = rd / name
        if f.is_file():
            try:
                out["tables"][name.replace(".json", "")] = json.loads(f.read_text())
            except Exception:
                pass
    for f in sorted(rd.glob("*.md")):
        out["docs"][f.stem] = f.read_text()
    hist = ROOT / "outputs" / "checkpoints" / "seg_mtl" / "history.json"
    if hist.is_file():
        out["tables"]["training_history"] = json.loads(hist.read_text())
    return _finite(out)


DIST = ROOT / "web" / "dist"

if DIST.is_dir():

    app.mount("/assets", StaticFiles(directory=str(DIST / "assets")), name="assets")
    if (DIST / "figures").is_dir():
        app.mount("/figures", StaticFiles(directory=str(DIST / "figures")), name="figures")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        """Serve index.html for any non-API path."""
        if full_path.startswith("api/"):
            raise HTTPException(404, "not found")
        candidate = DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST / "index.html")
else:
    @app.get("/")
    def placeholder():
        return JSONResponse({
            "message": "API is running. The frontend has not been built yet.",
            "build": "cd web && npm install && npm run build",
            "docs": "/docs",
        })
