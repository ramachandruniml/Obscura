"""Detector interface + RetinaFace + YOLOv8-face.

Always-run: Detection geometry, frame validation, the Detector.detect contract
(via a FakeDetector), and registry wiring.

Integration (auto-skip without weights / a face image): real inference on blank,
noise, and real-face frames; threshold monotonicity; odd frame sizes.
"""

from __future__ import annotations

import numpy as np
import pytest
from app.detectors import (
    VALID_BACKENDS,
    Detection,
    Detector,
    InvalidFrameError,
    get_detector_class,
    iou,
)
from app.detectors.base import validate_frame
from app.detectors.registry import build_detector


# --------------------------------------------------------------------------- #
# Detection geometry                                                           #
# --------------------------------------------------------------------------- #
class TestDetectionGeometry:
    def test_basic_dimensions(self) -> None:
        d = Detection(10, 20, 110, 220, 0.9)
        assert d.width == 100
        assert d.height == 200
        assert d.area == 20000
        assert d.center == (60, 120)
        assert d.as_int_box() == (10, 20, 110, 220)
        np.testing.assert_array_equal(d.xyxy(), np.array([10, 20, 110, 220], dtype=np.float32))

    def test_negative_size_is_clamped_to_zero(self) -> None:
        d = Detection(100, 100, 50, 50, 0.5)
        assert d.width == 0
        assert d.height == 0
        assert d.area == 0
        assert d.is_degenerate()

    def test_clip_to_bounds(self) -> None:
        d = Detection(-30, -10, 700, 500, 0.8).clip(640, 480)
        assert (d.x1, d.y1, d.x2, d.y2) == (0.0, 0.0, 640.0, 480.0)

    def test_expand_adds_padding_then_clips(self) -> None:
        d = Detection(100, 100, 200, 200, 0.8).expand(0.1, 640, 480)
        assert (d.x1, d.y1, d.x2, d.y2) == (90.0, 90.0, 210.0, 210.0)
        assert d.score == 0.8

    def test_expand_clips_at_frame_edge(self) -> None:
        d = Detection(0, 0, 100, 100, 0.8).expand(0.5, 640, 480)
        assert d.x1 == 0.0 and d.y1 == 0.0
        assert d.x2 == 150.0 and d.y2 == 150.0

    def test_landmarks_optional(self) -> None:
        assert Detection(0, 0, 10, 10, 0.5).landmarks is None
        lm = np.zeros((5, 2), dtype=np.float32)
        assert Detection(0, 0, 10, 10, 0.5, landmarks=lm).landmarks is lm


class TestIoU:
    def test_identical_boxes(self) -> None:
        d = Detection(0, 0, 10, 10, 1.0)
        assert iou(d, d) == pytest.approx(1.0)

    def test_disjoint_boxes(self) -> None:
        assert iou(Detection(0, 0, 10, 10, 1), Detection(20, 20, 30, 30, 1)) == 0.0

    def test_partial_overlap(self) -> None:
        a = Detection(0, 0, 10, 10, 1)
        b = Detection(5, 0, 15, 10, 1)
        # inter = 5*10 = 50 ; union = 100 + 100 - 50 = 150
        assert iou(a, b) == pytest.approx(50 / 150)

    def test_contained_box(self) -> None:
        a = Detection(0, 0, 10, 10, 1)
        b = Detection(2, 2, 8, 8, 1)
        assert iou(a, b) == pytest.approx(36 / 100)


# --------------------------------------------------------------------------- #
# Frame validation                                                             #
# --------------------------------------------------------------------------- #
class TestFrameValidation:
    def test_accepts_valid_bgr_frame(self) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        assert validate_frame(frame) is frame

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            [[1, 2, 3]],
            "not a frame",
            np.zeros((32, 32), dtype=np.uint8),  # grayscale, 2D
            np.zeros((32, 32, 4), dtype=np.uint8),  # RGBA
            np.zeros((32, 32, 1), dtype=np.uint8),  # single channel
            np.zeros((2, 32, 32, 3), dtype=np.uint8),  # batched
            np.zeros((32, 32, 3), dtype=np.float32),  # wrong dtype
            np.zeros((32, 32, 3), dtype=np.int64),  # wrong dtype
            np.zeros((1, 1, 3), dtype=np.uint8),  # too small
        ],
    )
    def test_rejects_bad_frames(self, bad: object) -> None:
        with pytest.raises(InvalidFrameError):
            validate_frame(bad)


# --------------------------------------------------------------------------- #
# Detector.detect contract (no weights)                                        #
# --------------------------------------------------------------------------- #
class FakeDetector(Detector):
    name = "fake"

    def __init__(self, canned: list[Detection], **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self.canned = canned
        self.calls = 0

    def _detect(self, frame: np.ndarray, confidence: float) -> list[Detection]:
        self.calls += 1
        return list(self.canned)


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


class TestDetectorContract:
    def test_sorts_by_score_desc(self, frame: np.ndarray) -> None:
        det = FakeDetector(
            [
                Detection(0, 0, 10, 10, 0.3),
                Detection(0, 0, 10, 10, 0.9),
                Detection(0, 0, 10, 10, 0.6),
            ]
        )
        scores = [d.score for d in det.detect(frame, 0.0)]
        assert scores == sorted(scores, reverse=True)

    def test_filters_below_threshold(self, frame: np.ndarray) -> None:
        det = FakeDetector([Detection(0, 0, 10, 10, 0.4), Detection(0, 0, 10, 10, 0.8)])
        assert len(det.detect(frame, 0.5)) == 1

    def test_drops_degenerate_boxes(self, frame: np.ndarray) -> None:
        det = FakeDetector([Detection(5, 5, 5, 5, 0.9), Detection(0, 0, 10, 10, 0.9)])
        assert len(det.detect(frame, 0.1)) == 1

    def test_caps_at_max_faces(self, frame: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.detectors.base.settings.max_faces", 3)
        det = FakeDetector([Detection(0, 0, 10, 10, 0.5 + i / 100) for i in range(10)])
        out = det.detect(frame, 0.0)
        assert len(out) == 3
        assert out[0].score == pytest.approx(0.59)  # highest kept

    def test_rejects_out_of_range_confidence(self, frame: np.ndarray) -> None:
        det = FakeDetector([])
        for bad in (-0.1, 1.1, 2.0):
            with pytest.raises(ValueError, match="confidence"):
                det.detect(frame, bad)

    def test_validates_frame_before_inference(self) -> None:
        det = FakeDetector([Detection(0, 0, 10, 10, 0.9)])
        with pytest.raises(InvalidFrameError):
            det.detect("nope")
        assert det.calls == 0

    def test_zero_faces_returns_empty_list(self, frame: np.ndarray) -> None:
        det = FakeDetector([])
        assert det.detect(frame, 0.5) == []

    def test_default_no_landmarks(self) -> None:
        assert FakeDetector([]).provides_landmarks is False


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #
class TestRegistry:
    def test_valid_backends(self) -> None:
        assert set(VALID_BACKENDS) == {"retinaface", "yolov8face"}

    def test_get_class_without_instantiation(self) -> None:
        assert get_detector_class("retinaface").__name__ == "RetinaFaceDetector"
        assert get_detector_class("yolov8face").__name__ == "YOLOv8FaceDetector"
        assert issubclass(get_detector_class("retinaface"), Detector)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown detector backend"):
            get_detector_class("deepface")

    def test_onnx_runtime_routes_to_onnx_detector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # wiring reaches OnnxFaceDetector; it only fails here because no .onnx exists
        monkeypatch.setattr("app.detectors.registry.settings.detector_runtime", "onnx")
        monkeypatch.setattr("app.config.settings.onnx_model_path", "weights/_absent_.onnx")
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            build_detector()


# --------------------------------------------------------------------------- #
# Real-inference integration (auto-skip without weights)                       #
# --------------------------------------------------------------------------- #
class TestInferenceIntegration:
    def test_blank_frame_has_no_faces(
        self, any_detector: Detector, blank_image: np.ndarray
    ) -> None:
        assert any_detector.detect(blank_image, 0.5) == []

    def test_noise_frame_has_no_confident_faces(
        self, any_detector: Detector, noise_image: np.ndarray
    ) -> None:
        assert any_detector.detect(noise_image, 0.6) == []

    def test_detects_real_face(self, any_detector: Detector, face_image: np.ndarray) -> None:
        dets = any_detector.detect(face_image, 0.5)
        assert len(dets) >= 1
        h, w = face_image.shape[:2]
        for d in dets:
            assert 0 <= d.x1 < d.x2 <= w
            assert 0 <= d.y1 < d.y2 <= h
            assert d.score >= 0.5

    def test_results_sorted_by_score(self, any_detector: Detector, face_image: np.ndarray) -> None:
        scores = [d.score for d in any_detector.detect(face_image, 0.3)]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_is_monotonic(self, any_detector: Detector, face_image: np.ndarray) -> None:
        low = len(any_detector.detect(face_image, 0.3))
        high = len(any_detector.detect(face_image, 0.9))
        assert high <= low

    def test_tiny_frame_does_not_crash(self, any_detector: Detector) -> None:
        out = any_detector.detect(np.zeros((8, 8, 3), dtype=np.uint8), 0.5)
        assert out == []

    def test_large_frame_does_not_crash(
        self, any_detector: Detector, rng: np.random.Generator
    ) -> None:
        big = rng.integers(0, 256, size=(1600, 2400, 3), dtype=np.uint8)
        assert isinstance(any_detector.detect(big, 0.7), list)


class TestRetinaFaceSpecifics:
    def test_module_names_match_biubug6_checkpoint_layout(self) -> None:
        """The vendored model's param names must equal the upstream checkpoint's,
        or load_state_dict silently loads a half-initialised model. (Regression:
        the SSH layers were renamed conv3X3 -> conv3x3, breaking every checkpoint.)
        """
        from app.detectors._retinaface import cfg_mnet
        from app.detectors._retinaface.model import RetinaFace

        keys = set(RetinaFace(cfg_mnet).state_dict())
        for ssh in ("ssh1", "ssh2", "ssh3"):
            for sub in ("conv3X3", "conv5X5_1", "conv5X5_2", "conv7X7_2", "conv7x7_3"):
                assert f"{ssh}.{sub}.0.weight" in keys, f"missing {ssh}.{sub}"
        assert "fpn.output1.0.weight" in keys
        assert "ClassHead.0.conv1x1.weight" in keys
        assert not any(".fc." in k or k.endswith(".fc.weight") for k in keys)  # classifier dropped

    def test_landmarks_shape(self, retinaface_detector: Detector, face_image: np.ndarray) -> None:
        dets = retinaface_detector.detect(face_image, 0.5)
        assert retinaface_detector.provides_landmarks
        for d in dets:
            assert d.landmarks is not None
            assert d.landmarks.shape == (5, 2)
