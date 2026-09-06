"""Detector selection and caching.

``get_detector_class`` is pure (no model I/O) — safe for tests and introspection.
``build_detector`` / ``get_cached_detector`` load weights; the worker uses the
cached one so a model is loaded once per process.
"""

from __future__ import annotations

from app.config import Settings, settings
from app.detectors.base import Detector
from app.logging import get_logger

log = get_logger(__name__)

_DETECTOR_CLASSES: dict[str, str] = {
    "retinaface": "app.detectors.retinaface:RetinaFaceDetector",
    "yolov8face": "app.detectors.yolov8_face:YOLOv8FaceDetector",
}
_ONNX_CLASS = "app.detectors.onnx_detector:OnnxFaceDetector"

VALID_BACKENDS = tuple(_DETECTOR_CLASSES)

_cache: dict[tuple[str, str, str], Detector] = {}


def _resolve(target: str) -> type[Detector]:
    module_name, _, attr = target.partition(":")
    module = __import__(module_name, fromlist=[attr])
    return getattr(module, attr)


def get_detector_class(backend: str) -> type[Detector]:
    """Resolve a backend name to its Detector class without instantiating it."""
    try:
        target = _DETECTOR_CLASSES[backend]
    except KeyError:
        raise ValueError(
            f"unknown detector backend {backend!r}; valid: {', '.join(VALID_BACKENDS)}"
        ) from None
    return _resolve(target)


def build_detector(cfg: Settings | None = None, **overrides: object) -> Detector:
    """Instantiate the configured detector (loads weights / the ONNX session)."""
    cfg = cfg or settings
    if cfg.detector_runtime == "onnx":
        detector = _resolve(_ONNX_CLASS)(**overrides)  # type: ignore[call-arg]
    else:
        detector = get_detector_class(cfg.detector_backend)(**overrides)  # type: ignore[call-arg]
    log.info("detector.built", backend=cfg.detector_backend, runtime=cfg.detector_runtime)
    return detector


def get_cached_detector(cfg: Settings | None = None) -> Detector:
    cfg = cfg or settings
    key = (cfg.detector_backend, cfg.detector_runtime, cfg.device)
    if key not in _cache:
        _cache[key] = build_detector(cfg)
        _cache[key].warmup()
    return _cache[key]


def clear_cache() -> None:
    _cache.clear()
