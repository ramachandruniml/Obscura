"""Redaction methods: Gaussian blur, pixelation, solid box.

Each :class:`Redactor` takes a frame and a list of regions (``Detection`` or any
object with a ``.box``), expands every region by ``pad_ratio`` so face edges are
not left exposed, clips to the frame, skips degenerate / off-frame boxes, and
applies the effect in place on the ROI. Pure image ops — no model, no state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from app.config import settings
from app.detectors.base import Detection
from app.logging import get_logger

log = get_logger(__name__)

REDACTION_METHODS: tuple[str, ...] = ("blur", "pixelate", "box")

_MAX_KERNEL = 199
_MIN_ROI = 3  # px on the short side; below this the region is skipped


def _as_box(item: object) -> Detection:
    box = getattr(item, "box", item)
    if not isinstance(box, Detection):
        raise TypeError(f"redaction region must be a Detection or have a .box, got {type(item)}")
    return box


def _odd(n: int) -> int:
    return n if n % 2 else n + 1


class Redactor(ABC):
    """Base class. Subclasses implement :meth:`_redact_roi` for a single ROI."""

    method: str = "abstract"

    def __init__(self, *, pad_ratio: float | None = None) -> None:
        self.pad_ratio = settings.box_padding_ratio if pad_ratio is None else float(pad_ratio)

    @abstractmethod
    def _redact_roi(self, roi: np.ndarray) -> np.ndarray:
        """Return a redacted version of ``roi`` (HxWx3 uint8). May return a new array."""

    def apply(
        self,
        frame: np.ndarray,
        regions: Iterable[object],
        *,
        pad_ratio: float | None = None,
        copy: bool = True,
    ) -> np.ndarray:
        """Redact every region in ``frame``. Returns the (by default copied) frame."""
        out = frame.copy() if copy else frame
        h, w = out.shape[:2]
        ratio = self.pad_ratio if pad_ratio is None else float(pad_ratio)

        applied = skipped = 0
        for item in regions:
            box = _as_box(item)
            box = box.expand(ratio, w, h) if ratio > 0 else box.clip(w, h)
            x1, y1, x2, y2 = box.as_int_box()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < _MIN_ROI or y2 - y1 < _MIN_ROI:
                skipped += 1
                continue
            out[y1:y2, x1:x2] = self._redact_roi(out[y1:y2, x1:x2])
            applied += 1

        log.debug("redaction.applied", method=self.method, regions=applied, skipped=skipped)
        return out


class GaussianBlurRedactor(Redactor):
    method = "blur"

    def __init__(
        self,
        *,
        pad_ratio: float | None = None,
        kernel_fraction: float | None = None,
        min_kernel: int | None = None,
        passes: int | None = None,
    ) -> None:
        super().__init__(pad_ratio=pad_ratio)
        self.kernel_fraction = (
            settings.blur_kernel_fraction if kernel_fraction is None else kernel_fraction
        )
        self.min_kernel = settings.blur_min_kernel if min_kernel is None else min_kernel
        self.passes = max(1, settings.blur_passes if passes is None else passes)

    def _redact_roi(self, roi: np.ndarray) -> np.ndarray:
        short = min(roi.shape[:2])
        k = int(round(short * self.kernel_fraction))
        k = max(k, self.min_kernel, 3)
        k = min(k, _MAX_KERNEL, short if short % 2 else short - 1)
        k = _odd(max(k, 3))
        blurred = roi
        for _ in range(self.passes):
            blurred = cv2.GaussianBlur(blurred, (k, k), 0)
        return blurred


class PixelateRedactor(Redactor):
    method = "pixelate"

    def __init__(self, *, pad_ratio: float | None = None, blocks: int | None = None) -> None:
        super().__init__(pad_ratio=pad_ratio)
        self.blocks = max(1, settings.pixelate_blocks if blocks is None else blocks)

    def _redact_roi(self, roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if w <= h:
            gw = min(w, self.blocks)
            gh = max(1, round(gw * h / w))
        else:
            gh = min(h, self.blocks)
            gw = max(1, round(gh * w / h))
        small = cv2.resize(roi, (gw, gh), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


class SolidBoxRedactor(Redactor):
    method = "box"

    def __init__(
        self, *, pad_ratio: float | None = None, color: Sequence[int] | None = None
    ) -> None:
        super().__init__(pad_ratio=pad_ratio)
        self.color = tuple(
            int(c) for c in (color if color is not None else settings.solid_box_color)
        )

    def _redact_roi(self, roi: np.ndarray) -> np.ndarray:
        roi[:] = self.color
        return roi


_REDACTORS: dict[str, type[Redactor]] = {
    "blur": GaussianBlurRedactor,
    "pixelate": PixelateRedactor,
    "box": SolidBoxRedactor,
}


def get_redactor_class(method: str) -> type[Redactor]:
    try:
        return _REDACTORS[method]
    except KeyError:
        raise ValueError(
            f"unknown redaction method {method!r}; valid: {', '.join(REDACTION_METHODS)}"
        ) from None


def build_redactor(method: str | None = None, **overrides: object) -> Redactor:
    method = method or settings.default_redaction
    return get_redactor_class(method)(**overrides)  # type: ignore[arg-type]
