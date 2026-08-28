"""Central configuration for the vision-based lane inference pipeline."""
from dataclasses import dataclass, field
from pathlib import Path


try:  # pragma: no cover - environment shim
    import os as _os, certifi as _certifi
    _os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
    _os.environ.setdefault("REQUESTS_CA_BUNDLE", _certifi.where())
except Exception:
    pass


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "datasets" / "idd_lite"
OUTPUT_ROOT = ROOT / "outputs"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"


NUM_CLASSES = 7
IGNORE_INDEX = 255

CLASS_NAMES = [
    "drivable",
    "non-drivable",
    "living-things",
    "vehicles",
    "roadside-objects",
    "far-objects",
    "sky",
]

DRIVABLE_ID = 0
VEHICLE_ID = 3
LIVING_ID = 2


CLASS_COLORS_BGR = [
    (60, 60, 220),
    (0, 165, 255),
    (255, 0, 255),
    (255, 128, 0),
    (255, 255, 0),
    (60, 200, 60),
    (240, 200, 160),
]


HSV_LOWER = (0, 0, 40)
HSV_UPPER = (180, 70, 210)
MORPH_KERNEL_SIZE = 5
BOTTOM_REGION_RATIO = 0.2
TRACK_START_RATIO = 0.6
SEARCH_MARGIN = 40
MIN_ROAD_COLUMNS = 50
MIN_SEGMENT_WIDTH = 20
ROW_STEP = 5


@dataclass
class TrainConfig:
    """Hyper-parameters for drivable-area / semantic segmentation training."""
    arch: str = "mobilenetv3_unet"
    input_size: tuple = (224, 320)
    batch_size: int = 16
    epochs: int = 80
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    label_smoothing: float = 0.0
    aux_loss_weight: float = 0.4
    dice_loss_weight: float = 0.5
    seed: int = 1337
    num_workers: int = 4
    amp: bool = False
    device: str = "auto"


@dataclass
class GeometryConfig:
    """Inverse-perspective-mapping and metric lane-inference parameters."""

    bev_width: int = 320
    bev_height: int = 480

    bev_lateral_m: float = 16.0
    bev_forward_m: float = 30.0
    bev_near_m: float = 4.0


    camera_height_m: float = 1.75
    fov_deg: float = 90.0


    vp_smooth_alpha: float = 0.15
    vp_min_road_rows: int = 8

    lane_width_urban_m: float = 3.50
    lane_width_rural_m: float = 3.00
    min_lane_width_m: float = 2.50
    max_lanes: int = 8


@dataclass
class TrackerConfig:
    """Kalman smoothing of inferred lane geometry across video frames."""
    process_noise: float = 1e-3
    measurement_noise: float = 5e-2
    max_missed_frames: int = 8
    init_uncertainty: float = 1.0


@dataclass
class FusionConfig:
    """Confidence-weighted CV+DL fusion (replaces the Phase 3 bitwise-OR)."""
    dl_weight: float = 0.75
    cv_weight: float = 0.25
    decision_threshold: float = 0.5
    dl_trust_floor: float = 0.35
    min_blob_area_ratio: float = 0.01


@dataclass
class PipelineConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)


DEFAULT = PipelineConfig()


CHECKPOINT_PREFERENCE = ("seg_idd20k", "seg_idd640", "seg_mtl", "seg_mnv3")


def default_checkpoint() -> Path:
    """First available checkpoint in preference order (last entry if none)."""
    paths = [CHECKPOINT_ROOT / name / "best.pt" for name in CHECKPOINT_PREFERENCE]
    return next((p for p in paths if p.is_file()), paths[-1])


def resolve_device(preference: str = "auto") -> str:
    """Pick the fastest available torch backend."""
    import torch

    if preference != "auto":
        return preference
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
