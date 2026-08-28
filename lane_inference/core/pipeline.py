"""End-to-end lane inference pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import config as cfg
from core.asphalt_detector import detect_asphalt, keep_bottom_connected
from core.dl_drivable import SegmentationResult, fuse_masks, gated_fallback
from core.geometry import (IPMTransform, VanishingPoint, VanishingPointTracker,
                           default_vanishing_point, fuse_vanishing_points,
                           vp_from_line_segments, vp_from_road_mask)
from core.lane_tracker import (LaneCountStabiliser, LaneKalmanFilter, LaneModel,
                               infer_lanes)
from core.lane_trajectory import (EmpiricalLanes, TrajectoryLaneEstimator,
                                  combine as combine_lane_sources)
from core.semantics import (ScenePlausibility, TrafficState, WrongSideDetector,
                            analyse_traffic, assess_scene, extract_road_users)
from utils.image_utils import fill_holes, remove_small_components, smooth_mask


@dataclass
class PipelineOptions:
    use_dl: bool = True
    use_cv: bool = True
    use_fusion: bool = False

    use_gated_fallback: bool = True
    use_cleanup: bool = True
    use_ipm: bool = True
    use_markings: bool = True
    use_occlusion_bridging: bool = True
    use_temporal: bool = False
    use_line_vp_refinement: bool = True
    use_trajectory_lanes: bool = False

    use_analytics: bool = True
    check_scene: bool = True
    detect_wrong_side: bool = False
    bev_max_y: float | None = 22.0


@dataclass
class PipelineResult:
    image: np.ndarray
    road_mask: np.ndarray
    road_confidence: np.ndarray
    semantic: SegmentationResult | None
    vanishing_point: VanishingPoint | None
    ipm: IPMTransform | None
    lane_model: LaneModel | None
    timings: dict = field(default_factory=dict)
    road_users: list = field(default_factory=list)
    traffic: TrafficState | None = None
    empirical_lanes: EmpiricalLanes | None = None
    lane_source: str = "geometric"
    scene: ScenePlausibility | None = None
    wrong_side: list = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return float(sum(self.timings.values()))

    @property
    def fps(self) -> float:
        t = self.total_ms
        return 1000.0 / t if t > 0 else float("nan")


class LaneInferencePipeline:
    """Stateful pipeline. Call `reset()` between independent video sequences."""

    def __init__(self, backend=None, options: PipelineOptions | None = None,
                 geom: "cfg.GeometryConfig" = None,
                 fusion: "cfg.FusionConfig" = None):
        self.backend = backend
        self.opt = options or PipelineOptions()
        self.geom = geom or cfg.DEFAULT.geometry
        self.fusion_cfg = fusion or cfg.DEFAULT.fusion
        self.vp_tracker = VanishingPointTracker()
        self.lane_filter = LaneKalmanFilter()
        self.lane_votes = LaneCountStabiliser()
        self.trajectory = TrajectoryLaneEstimator()
        self.wrong_side = WrongSideDetector()
        if self.backend is None:
            self.opt.use_dl = False

    def reset(self) -> None:
        self.vp_tracker.reset()
        self.lane_filter.reset()
        self.lane_votes.reset()
        self.trajectory.reset()
        self.wrong_side.reset()


    def _road_estimates(self, image):
        t = {}
        semantic = None
        dl_conf = None
        cv_conf = None

        if self.opt.use_dl and self.backend is not None:
            t0 = time.perf_counter()
            semantic = self.backend(image)
            dl_conf = semantic.drivable_confidence
            t["segmentation"] = (time.perf_counter() - t0) * 1000.0

        if self.opt.use_cv:
            t0 = time.perf_counter()
            _, cv_conf = detect_asphalt(image, return_confidence=True)
            t["classical"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self.last_branch = "learned"
        if dl_conf is not None and cv_conf is not None and self.opt.use_fusion:
            mask, conf = fuse_masks(cv_conf, dl_conf, self.fusion_cfg)
        elif dl_conf is not None and cv_conf is not None and self.opt.use_gated_fallback:
            mask, conf, self.last_branch = gated_fallback(cv_conf, dl_conf,
                                                          self.fusion_cfg)
        elif dl_conf is not None:
            conf = dl_conf
            mask = (conf > 0.5).astype(np.uint8) * 255
        elif cv_conf is not None:
            conf = cv_conf
            mask = (conf > 0.35).astype(np.uint8) * 255
        else:
            raise RuntimeError("Both the DL and CV branches are disabled.")
        t["fusion"] = (time.perf_counter() - t0) * 1000.0

        if self.opt.use_cleanup:
            t0 = time.perf_counter()
            mask = smooth_mask(mask, 5)
            mask = remove_small_components(mask, self.fusion_cfg.min_blob_area_ratio)


            mask = keep_bottom_connected(mask)
            mask = fill_holes(mask, max_hole_ratio=0.15)
            t["cleanup"] = (time.perf_counter() - t0) * 1000.0

        return mask, conf, semantic, t


    def _geometry(self, image, mask):
        t0 = time.perf_counter()
        vp = vp_from_road_mask(mask)
        if self.opt.use_line_vp_refinement:
            vp = fuse_vanishing_points(vp, vp_from_line_segments(image, mask))
        if self.opt.use_temporal:
            vp = self.vp_tracker.update(vp)
        if vp is None:
            vp = default_vanishing_point(mask.shape)
        ipm = IPMTransform.from_vanishing_point(vp, mask.shape[:2], self.geom)
        return vp, ipm, (time.perf_counter() - t0) * 1000.0


    def process(self, image: np.ndarray) -> PipelineResult:
        mask, conf, semantic, timings = self._road_estimates(image)


        scene = None
        if semantic is not None and self.opt.check_scene:
            scene = assess_scene(semantic.labels)

        vp = ipm = lane_model = None
        if self.opt.use_ipm:
            vp, ipm, dt = self._geometry(image, mask)
            timings["geometry"] = dt

            t0 = time.perf_counter()
            bev_mask = ipm.warp_mask(mask)
            bev_image = ipm.warp_image(image) if self.opt.use_markings else None


            bev_occ = None
            if semantic is not None and self.opt.use_occlusion_bridging:
                occ = ((semantic.labels == cfg.VEHICLE_ID) |
                       (semantic.labels == cfg.LIVING_ID)).astype(np.uint8) * 255
                bev_occ = ipm.warp_mask(occ)

            bev_lane = None
            if semantic is not None and semantic.lane_prob is not None:
                lm = semantic.lane_mask()
                if lm is not None:
                    bev_lane = ipm.warp_mask(lm)


            if scene is not None and not scene:
                lane_model = None
            else:
                lane_model = infer_lanes(
                    bev_mask, ipm, bev_image, self.geom,
                    use_markings=self.opt.use_markings,
                    bev_occluders=bev_occ,
                    bev_lane=bev_lane,
                    vp_confidence=vp.confidence if vp else 0.5)
            if self.opt.use_temporal:
                self.lane_filter.update(lane_model)
                if lane_model is not None:
                    lane_model = self.lane_filter.apply(lane_model)
                    voted = self.lane_votes.update(lane_model.n_lanes)
                    if voted is not None and voted != lane_model.n_lanes:
                        lane_model.n_lanes = voted
                        lane_model.boundary_offsets = np.linspace(
                            0.0, lane_model.carriageway_width_m, voted + 1)
                        lane_model.lane_width_m = (
                            lane_model.carriageway_width_m / max(voted, 1))
                else:
                    self.lane_votes.update(None)
            timings["lane_inference"] = (time.perf_counter() - t0) * 1000.0


        users, traffic, empirical, source, flagged = [], None, None, "geometric", []
        if ipm is not None and semantic is not None and self.opt.use_analytics:
            t0 = time.perf_counter()
            users = extract_road_users(semantic.labels, ipm, lane_model)


            if self.opt.use_trajectory_lanes and self.opt.use_temporal:
                self.trajectory.add_frame(users, lane_model, ipm)
                empirical = self.trajectory.estimate()
                lane_model, source = combine_lane_sources(lane_model, empirical)
                if lane_model is not None:
                    users = extract_road_users(semantic.labels, ipm, lane_model)

            traffic = analyse_traffic(users, lane_model, ipm)
            if self.opt.detect_wrong_side and self.opt.use_temporal:
                flagged = self.wrong_side.update(users, lane_model)
            timings["analytics"] = (time.perf_counter() - t0) * 1000.0

        return PipelineResult(image=image, road_mask=mask, road_confidence=conf,
                              semantic=semantic, vanishing_point=vp, ipm=ipm,
                              lane_model=lane_model, timings=timings,
                              road_users=users, traffic=traffic,
                              empirical_lanes=empirical, lane_source=source,
                              wrong_side=flagged, scene=scene)
