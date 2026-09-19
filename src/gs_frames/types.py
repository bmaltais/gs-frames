from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

SharpnessMethod = Literal["tenengrad", "laplacian", "combined"]
OverlapMetric = Literal["none", "flow", "orb", "homography"]
SelectMode = Literal["time", "flow", "overlap-greedy", "overlap-beam"]
ImageFormat = Literal["jpg", "png"]
RotationMode = Literal["auto", "none", "90", "180", "270"]
RotationSource = Literal["container_meta", "ffprobe", "user_override", "none_detected"]


@dataclass
class FrameScore:
    index: int
    timestamp_s: float
    sharpness: float
    sharpness_tenengrad: float
    sharpness_laplacian: float
    flow_median: float = 0.0
    flow_accum_from_prev_keep: float = 0.0


@dataclass
class Selection:
    index: int
    timestamp_s: float
    sharpness: float
    overlap_with_prev: Optional[float]
    overlap_metric: str
    reason: str


class ConfigError(ValueError):
    """Raised when an ExtractConfig fails validation."""


@dataclass
class ExtractConfig:
    video: Path
    output_dir: Path
    overlap_min: float = 0.70
    overlap_max: float = 0.80
    target_overlap: Optional[float] = None
    sharpness: SharpnessMethod = "combined"
    overlap_metric: OverlapMetric = "orb"
    mode: SelectMode = "overlap-greedy"
    max_frames: Optional[int] = None
    min_sharpness: Optional[float] = None
    min_sharpness_percentile: float = 5.0
    analysis_scale: float = 0.25
    analysis_max_width: int = 640
    every_seconds: Optional[float] = None
    chunk_frames: Optional[int] = None
    format: ImageFormat = "jpg"
    jpeg_quality: int = 95
    preview: bool = False
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    workers: int = 1
    seed_policy: str = "first-usable"
    flow_cell_size: int = 16
    orb_nfeatures: int = 1500
    match_ratio: float = 0.75
    ransac_reproj_threshold: float = 3.0
    search_expand_frames: int = 60
    beam_width: int = 5
    overlap_penalty_lambda: float = 4.0
    rotation: RotationMode = "auto"
    use_container_timestamps: bool = True

    def __post_init__(self) -> None:
        if self.target_overlap is None:
            self.target_overlap = (self.overlap_min + self.overlap_max) / 2

        if not (0 <= self.overlap_min < self.overlap_max <= 1):
            raise ConfigError(
                f"overlap range must satisfy 0 <= min < max <= 1, "
                f"got min={self.overlap_min}, max={self.overlap_max}"
            )
        if not (0 < self.analysis_scale <= 1):
            raise ConfigError(f"analysis_scale must be in (0, 1], got {self.analysis_scale}")
        if not (1 <= self.jpeg_quality <= 100):
            raise ConfigError(f"jpeg_quality must be in [1, 100], got {self.jpeg_quality}")
        if self.start_s is not None and self.end_s is not None and self.start_s >= self.end_s:
            raise ConfigError(f"start_s ({self.start_s}) must be < end_s ({self.end_s})")
        if (
            self.start_frame is not None
            and self.end_frame is not None
            and self.start_frame >= self.end_frame
        ):
            raise ConfigError(
                f"start_frame ({self.start_frame}) must be < end_frame ({self.end_frame})"
            )
        if not self.video.exists() or not self.video.is_file():
            raise ConfigError(f"video not found: {self.video}")
