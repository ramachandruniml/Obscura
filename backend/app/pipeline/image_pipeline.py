"""Image redaction pipeline: decode -> detect -> redact -> encode.

Frames live only in memory for the duration of the call. Nothing is persisted
except the encoded output written to ``dst``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.detectors.base import Detector
from app.errors import CorruptedMediaError
from app.logging import get_logger
from app.redaction.redactors import Redactor

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ImageResult:
    n_faces: int
    width: int
    height: int


def _imdecode(path: Path) -> np.ndarray:
    # np.fromfile handles non-ASCII paths on Windows; cv2.imread does not.
    buf = np.fromfile(str(path), dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    if frame is None:
        raise CorruptedMediaError(f"could not decode image: {path.name}")
    return frame


def redact_image(
    src: Path,
    dst: Path,
    detector: Detector,
    redactor: Redactor,
    confidence: float,
) -> ImageResult:
    log.info("pipeline.image.start", src=src.name, dst=dst.name)
    frame = _imdecode(src)
    h, w = frame.shape[:2]

    log.info("detection.started", width=w, height=h, detector=detector.name)
    detections = detector.detect(frame, confidence)
    log.info("detection.completed", n_faces=len(detections))

    redacted = redactor.apply(frame, detections, copy=True)
    log.info("redaction.applied", method=redactor.method, n_regions=len(detections))

    ext = dst.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(ext, redacted)
    if not ok:
        ok, encoded = cv2.imencode(".png", redacted)
        dst = dst.with_suffix(".png")
    encoded.tofile(str(dst))

    log.info("job.completed", kind="image", n_faces=len(detections))
    return ImageResult(n_faces=len(detections), width=w, height=h)
