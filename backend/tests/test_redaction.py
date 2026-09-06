"""Redaction methods (blur / pixelate / box).

Pure image ops — no model weights. Pixel-level checks that the region is
actually destroyed, that everything outside the padded box is byte-for-byte
untouched, and that degenerate / off-frame / tiny boxes never crash.
"""

from __future__ import annotations

import numpy as np
import pytest
from app.detectors.base import Detection
from app.redaction import (
    REDACTION_METHODS,
    GaussianBlurRedactor,
    PixelateRedactor,
    Redactor,
    SolidBoxRedactor,
    build_redactor,
    get_redactor_class,
)
from app.tracking import TrackedFace

BOX = Detection(60, 50, 140, 150, 0.99)  # well inside a 200x200 frame
SQUARE = Detection(60, 60, 140, 140, 0.99)


def checker(h: int = 200, w: int = 200, cell: int = 4) -> np.ndarray:
    """High-frequency BGR checkerboard so blur/pixelate have detail to destroy."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = ((((yy // cell) + (xx // cell)) % 2) * 255).astype(np.uint8)
    return np.stack([base, base, base], axis=-1)


@pytest.fixture
def frame() -> np.ndarray:
    return checker()


def changed_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.any(before != after, axis=2)


# --------------------------------------------------------------------------- #
# Region is destroyed / surroundings are not                                   #
# --------------------------------------------------------------------------- #
class TestCoreEffect:
    @pytest.mark.parametrize("method", REDACTION_METHODS)
    def test_inside_box_is_changed(self, frame: np.ndarray, method: str) -> None:
        out = build_redactor(method).apply(frame, [BOX], pad_ratio=0)
        x1, y1, x2, y2 = BOX.as_int_box()
        assert not np.array_equal(out[y1:y2, x1:x2], frame[y1:y2, x1:x2])

    @pytest.mark.parametrize("method", REDACTION_METHODS)
    def test_outside_box_is_byte_identical(self, frame: np.ndarray, method: str) -> None:
        out = build_redactor(method).apply(frame, [BOX], pad_ratio=0)
        x1, y1, x2, y2 = BOX.as_int_box()
        mask = np.ones(frame.shape[:2], dtype=bool)
        mask[y1:y2, x1:x2] = False
        np.testing.assert_array_equal(out[mask], frame[mask])

    @pytest.mark.parametrize("method", REDACTION_METHODS)
    def test_shape_and_dtype_preserved(self, frame: np.ndarray, method: str) -> None:
        out = build_redactor(method).apply(frame, [BOX])
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_blur_destroys_high_frequency_detail(self, frame: np.ndarray) -> None:
        out = GaussianBlurRedactor().apply(frame, [BOX], pad_ratio=0)
        x1, y1, x2, y2 = BOX.as_int_box()
        before = frame[y1:y2, x1:x2].std()
        after = out[y1:y2, x1:x2].std()
        assert after < before * 0.5

    def test_pixelate_bounds_unique_colors(self, frame: np.ndarray) -> None:
        blocks = 8
        out = PixelateRedactor(blocks=blocks).apply(frame, [SQUARE], pad_ratio=0)
        x1, y1, x2, y2 = SQUARE.as_int_box()
        roi = out[y1:y2, x1:x2].reshape(-1, 3)
        assert len(np.unique(roi, axis=0)) <= blocks * blocks

    def test_pixelate_output_is_a_block_grid(self, frame: np.ndarray) -> None:
        blocks = 6
        out = PixelateRedactor(blocks=blocks).apply(frame, [SQUARE], pad_ratio=0)
        x1, y1, x2, y2 = SQUARE.as_int_box()
        roi = out[y1:y2, x1:x2]
        # INTER_NEAREST upscaling from a ~blocks x blocks grid => at most `blocks`
        # distinct row patterns and `blocks` distinct column patterns
        rows = np.unique(roi.reshape(roi.shape[0], -1), axis=0)
        cols = np.unique(roi.transpose(1, 0, 2).reshape(roi.shape[1], -1), axis=0)
        assert len(rows) <= blocks
        assert len(cols) <= blocks

    def test_solid_box_fills_exact_color(self, frame: np.ndarray) -> None:
        out = SolidBoxRedactor(color=(10, 20, 30)).apply(frame, [BOX], pad_ratio=0)
        x1, y1, x2, y2 = BOX.as_int_box()
        assert np.all(out[y1:y2, x1:x2] == (10, 20, 30))

    def test_solid_box_defaults_to_black(self, frame: np.ndarray) -> None:
        out = build_redactor("box").apply(frame, [BOX], pad_ratio=0)
        x1, y1, x2, y2 = BOX.as_int_box()
        assert np.all(out[y1:y2, x1:x2] == 0)


# --------------------------------------------------------------------------- #
# Padding                                                                      #
# --------------------------------------------------------------------------- #
class TestPadding:
    def test_padding_grows_and_contains_the_tight_region(self, frame: np.ndarray) -> None:
        r = build_redactor("box")
        tight = changed_mask(frame, r.apply(frame, [BOX], pad_ratio=0))
        padded = changed_mask(frame, r.apply(frame, [BOX], pad_ratio=0.4))
        assert padded.sum() > tight.sum()
        assert np.all(padded[tight])  # padded area fully covers the tight area

    def test_default_pad_ratio_comes_from_settings(self, frame: np.ndarray) -> None:
        r = build_redactor("box")
        default = changed_mask(frame, r.apply(frame, [BOX]))
        zero = changed_mask(frame, r.apply(frame, [BOX], pad_ratio=0))
        assert default.sum() > zero.sum()


# --------------------------------------------------------------------------- #
# Zero faces / immutability                                                    #
# --------------------------------------------------------------------------- #
class TestNoOpCases:
    @pytest.mark.parametrize("method", REDACTION_METHODS)
    def test_no_regions_returns_unchanged_copy(self, frame: np.ndarray, method: str) -> None:
        out = build_redactor(method).apply(frame, [])
        np.testing.assert_array_equal(out, frame)
        assert out is not frame

    def test_copy_false_mutates_and_returns_same_array(self, frame: np.ndarray) -> None:
        out = build_redactor("box").apply(frame, [BOX], pad_ratio=0, copy=False)
        assert out is frame

    def test_original_not_mutated_when_copy_true(self, frame: np.ndarray) -> None:
        snapshot = frame.copy()
        build_redactor("box").apply(frame, [BOX])
        np.testing.assert_array_equal(frame, snapshot)


# --------------------------------------------------------------------------- #
# Degenerate / off-frame boxes                                                 #
# --------------------------------------------------------------------------- #
class TestEdgeBoxes:
    def test_box_partly_outside_frame_is_clipped(self, frame: np.ndarray) -> None:
        out = build_redactor("box").apply(frame, [Detection(-40, -40, 60, 60, 0.9)], pad_ratio=0)
        assert np.all(out[0:60, 0:60] == 0)
        np.testing.assert_array_equal(out[60:, 60:], frame[60:, 60:])

    @pytest.mark.parametrize(
        "box",
        [
            Detection(300, 300, 380, 380, 0.9),  # entirely past the frame
            Detection(-90, -90, -10, -10, 0.9),  # entirely before the frame
        ],
    )
    def test_box_fully_outside_frame_is_skipped(self, frame: np.ndarray, box: Detection) -> None:
        out = build_redactor("blur").apply(frame, [box])
        np.testing.assert_array_equal(out, frame)

    @pytest.mark.parametrize(
        "box",
        [
            Detection(50, 50, 50, 120, 0.9),  # zero width
            Detection(50, 50, 52, 52, 0.9),  # below the 3px minimum
            Detection(80, 80, 79, 79, 0.9),  # inverted
        ],
    )
    def test_degenerate_box_is_skipped(self, frame: np.ndarray, box: Detection) -> None:
        out = build_redactor("pixelate").apply(frame, [box], pad_ratio=0)
        np.testing.assert_array_equal(out, frame)

    @pytest.mark.parametrize("method", REDACTION_METHODS)
    def test_tiny_valid_box_does_not_crash(self, frame: np.ndarray, method: str) -> None:
        out = build_redactor(method).apply(frame, [Detection(50, 50, 58, 58, 0.9)], pad_ratio=0)
        assert out.shape == frame.shape

    def test_odd_sized_frame(self) -> None:
        odd = checker(h=97, w=53, cell=3)
        out = build_redactor("blur").apply(odd, [Detection(10, 10, 45, 80, 0.9)], pad_ratio=0.2)
        assert out.shape == odd.shape


# --------------------------------------------------------------------------- #
# Multiple regions / input types                                               #
# --------------------------------------------------------------------------- #
class TestRegions:
    def test_multiple_overlapping_boxes_all_covered(self, frame: np.ndarray) -> None:
        boxes = [
            Detection(20, 20, 120, 120, 0.9),
            Detection(80, 80, 180, 180, 0.9),
            Detection(10, 150, 60, 195, 0.9),
        ]
        out = build_redactor("box").apply(frame, boxes, pad_ratio=0)
        for b in boxes:
            x1, y1, x2, y2 = b.as_int_box()
            assert np.all(out[y1:y2, x1:x2] == 0)

    def test_accepts_tracked_face(self, frame: np.ndarray) -> None:
        tf = TrackedFace(track_id=3, box=BOX, coasting=True)
        out = build_redactor("box").apply(frame, [tf], pad_ratio=0)
        x1, y1, x2, y2 = BOX.as_int_box()
        assert np.all(out[y1:y2, x1:x2] == 0)

    def test_rejects_unsupported_region_type(self, frame: np.ndarray) -> None:
        with pytest.raises(TypeError):
            build_redactor("box").apply(frame, [(1, 2, 3, 4)])


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #
class TestRegistry:
    def test_method_names(self) -> None:
        assert REDACTION_METHODS == ("blur", "pixelate", "box")

    @pytest.mark.parametrize(
        ("method", "cls"),
        [
            ("blur", GaussianBlurRedactor),
            ("pixelate", PixelateRedactor),
            ("box", SolidBoxRedactor),
        ],
    )
    def test_get_class(self, method: str, cls: type) -> None:
        assert get_redactor_class(method) is cls
        assert issubclass(cls, Redactor)

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown redaction method"):
            get_redactor_class("scramble")

    def test_build_defaults_to_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.redaction.redactors.settings.default_redaction", "pixelate")
        assert isinstance(build_redactor(), PixelateRedactor)

    def test_build_passes_overrides(self) -> None:
        r = build_redactor("pixelate", blocks=3)
        assert isinstance(r, PixelateRedactor)
        assert r.blocks == 3
