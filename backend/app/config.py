"""Single source of truth for every adjustable knob.

All values are overridable via environment variables (see ``.env.example``).
Downstream modules import ``settings`` from here and never hardcode limits,
thresholds, paths, or backend choices.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Environment ---------------------------------------------------------
    obscura_env: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False

    # --- API --------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"]
    )

    # --- Redis / Celery ---------------------------------------------------
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"
    celery_task_always_eager: bool = False  # True -> run jobs inline (tests / no broker)

    # --- Pipeline / output ------------------------------------------------
    delete_input_on_success: bool = True
    video_output_codec: str = "libx264"
    video_output_crf: int = 23
    video_output_pix_fmt: str = "yuv420p"

    # --- Storage / retention -------------------------------------------------
    storage_dir: Path = Path("/data")
    result_ttl_seconds: int = 3600
    cleanup_interval_seconds: int = 300

    # --- Upload limits ----------------------------------------------------
    max_upload_bytes: int = 100 * 1024 * 1024
    allowed_image_types: list[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp"]
    )
    allowed_video_types: list[str] = Field(
        default_factory=lambda: ["video/mp4", "video/quicktime", "video/webm"]
    )
    max_video_duration_seconds: float = 300.0
    max_image_pixels: int = 50_000_000

    # --- Detection ------------------------------------------------------
    detector_backend: Literal["retinaface", "yolov8face"] = "retinaface"
    detector_runtime: Literal["pytorch", "onnx"] = "pytorch"
    device: Literal["cpu", "cuda"] = "cpu"
    retinaface_weights: Path = Path("weights/retinaface_mobilenet0.25.pth")
    retinaface_backbone: Literal["mobile0.25", "resnet50"] = "mobile0.25"
    retinaface_input_size: int = 640
    retinaface_max_side: int = 2048
    retinaface_top_k: int = 5000
    retinaface_keep_top_k: int = 750
    yolov8_face_weights: Path = Path("weights/yolov8n-face.pt")
    yolov8_input_size: int = 640
    onnx_model_path: Path = Path("weights/detector.onnx")
    onnx_opset: int = 12
    default_confidence: float = 0.5
    nms_iou: float = 0.4
    max_faces: int = 500

    # --- Redaction --------------------------------------------------------
    default_redaction: Literal["blur", "pixelate", "box"] = "blur"
    box_padding_ratio: float = 0.15
    blur_kernel_fraction: float = 0.25
    blur_min_kernel: int = 15
    blur_passes: int = 2
    pixelate_blocks: int = 12
    solid_box_color: tuple[int, int, int] = (0, 0, 0)  # BGR

    # --- Video tracking -----------------------------------------------------
    detect_every_n_frames: int = 5
    track_activation_threshold: float = 0.25
    min_matching_threshold: float = 0.8
    track_buffer_frames: int = 30
    track_lost_padding_ratio: float = 0.30
    track_fps_default: int = 30
    track_max_coast_frames: int = 0  # <= 0 -> auto (3 * detect_every_n_frames)

    @field_validator("cors_origins", "allowed_image_types", "allowed_video_types", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("solid_box_color", mode="before")
    @classmethod
    def _parse_bgr(cls, v: object) -> object:
        if isinstance(v, str):
            parts = [int(p) for p in v.replace("(", "").replace(")", "").split(",") if p.strip()]
            return tuple(parts)
        return v

    @property
    def allowed_types(self) -> list[str]:
        return [*self.allowed_image_types, *self.allowed_video_types]

    def kind_for_mime(self, mime: str) -> Literal["image", "video"] | None:
        if mime in self.allowed_image_types:
            return "image"
        if mime in self.allowed_video_types:
            return "video"
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
