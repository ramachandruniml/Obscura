"""Common face-detection interface.

Everything downstream (tracking, redaction, pipeline, benchmark) depends only on
this module — never on a concrete detector. A detector consumes raw BGR frames
(``np.uint8``, ``HxWx3``) and returns ``Detection`` objects in absolute pixel
coordinates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np

from app.config import settings
from app.logging import get_logger

log = get_logger(__name__)


class InvalidFrameError(ValueError):
    """Raised when a frame passed to ``Detector.detect`` is not a valid image array."""


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detected face in absolute pixel coordinates (xyxy).

    ``landmarks`` is an optional ``(5, 2)`` float array in the order
    [left_eye, right_eye, nose, mouth_left, mouth_right]. Detectors that do not
    produce landmarks set it to ``None``; that is still a conformant detection.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    landmarks: np.ndarray | None = None

    # -- geometry ---------------------------------------------------------------
    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def as_int_box(self) -> tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))

    def clip(self, width: int, height: int) -> Detection:
        """Return a copy clamped to the frame bounds."""
        x1 = min(max(self.x1, 0.0), float(width))
        y1 = min(max(self.y1, 0.0), float(height))
        x2 = min(max(self.x2, 0.0), float(width))
        y2 = min(max(self.y2, 0.0), float(height))
        return replace(self, x1=x1, y1=y1, x2=x2, y2=y2)

    def expand(self, ratio: float, width: int, height: int) -> Detection:
        """Pad the box by ``ratio`` of its size on every side, then clip.

        Used by the redaction module so face edges are not left exposed.
        """
        dx = self.width * ratio
        dy = self.height * ratio
        return replace(
            self,
            x1=self.x1 - dx,
            y1=self.y1 - dy,
            x2=self.x2 + dx,
            y2=self.y2 + dy,
        ).clip(width, height)

    def is_degenerate(self, min_size: float = 1.0) -> bool:
        return self.width < min_size or self.height < min_size


def iou(a: Detection, b: Detection) -> float:
    """Intersection-over-union of two detections' boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def validate_frame(frame: object) -> np.ndarray:
    """Return ``frame`` as a validated ``HxWx3`` uint8 array or raise ``InvalidFrameError``."""
    if not isinstance(frame, np.ndarray):
        raise InvalidFrameError(f"expected numpy.ndarray, got {type(frame).__name__}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise InvalidFrameError(f"expected HxWx3 BGR frame, got shape {frame.shape}")
    if frame.dtype != np.uint8:
        raise InvalidFrameError(f"expected uint8 frame, got dtype {frame.dtype}")
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise InvalidFrameError(f"frame too small: {frame.shape[:2]}")
    return frame


class Detector(ABC):
    """Abstract face detector.

    Implementations override :meth:`_detect` (and optionally :meth:`warmup`).
    Callers use :meth:`detect`, which validates the frame, applies the score
    threshold contract, sorts by score, and caps the result at
    ``settings.max_faces``.
    """

    #: Stable identifier used by the registry, logs, and benchmark reports.
    name: str = "abstract"
    #: Whether :meth:`_detect` populates ``Detection.landmarks``.
    provides_landmarks: bool = False

    def __init__(
        self, *, default_confidence: float | None = None, device: str | None = None
    ) -> None:
        self.default_confidence = (
            settings.default_confidence if default_confidence is None else default_confidence
        )
        self.device = device or settings.device

    @abstractmethod
    def _detect(self, frame: np.ndarray, confidence: float) -> list[Detection]:
        """Run inference on a validated BGR frame. Return detections at/above ``confidence``."""

    def warmup(self) -> None:  # noqa: B027  # optional hook, intentionally non-abstract
        """Optional: run a dummy inference so the first real call is not slow.

        Concrete detectors override this; the default is a deliberate no-op so
        callers can always invoke it.
        """

    def detect(self, frame: object, confidence: float | None = None) -> list[Detection]:
        conf = self.default_confidence if confidence is None else float(confidence)
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {conf}")
        arr = validate_frame(frame)

        dets = self._detect(arr, conf)
        dets = [d for d in dets if d.score >= conf and not d.is_degenerate()]
        dets.sort(key=lambda d: d.score, reverse=True)
        if len(dets) > settings.max_faces:
            log.warning(
                "detection.capped",
                detector=self.name,
                found=len(dets),
                cap=settings.max_faces,
            )
            dets = dets[: settings.max_faces]
        return dets

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} device={self.device!r}>"
