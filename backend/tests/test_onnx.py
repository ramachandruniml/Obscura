"""ONNX path: YOLO pre/post-processing, latency helpers, registry wiring.

The decode / letterbox / NMS code is pure numpy and fully tested here without a
model. A real end-to-end OnnxFaceDetector test is weight-gated (skips without
`weights/detector.onnx`).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from app.benchmarking import LatencyStats, measure_latency, render_latency_report
from app.detectors._yolo_onnx import (
    decode_yolo_output,
    keypoint_count,
    letterbox,
    nms_numpy,
)
from app.detectors.base import Detection


# --------------------------------------------------------------------------- #
# letterbox                                                                    #
# --------------------------------------------------------------------------- #
class TestLetterbox:
    def test_output_is_square_and_ratio_correct(self) -> None:
        img = np.zeros((90, 160, 3), dtype=np.uint8)
        out, ratio, (pad_l, pad_t) = letterbox(img, 640)
        assert out.shape == (640, 640, 3)
        assert ratio == pytest.approx(640 / 160)
        assert pad_l == 0  # width is the limiting dimension
        assert pad_t == (640 - round(90 * ratio)) // 2

    def test_unmap_round_trips(self) -> None:
        img = np.zeros((120, 200, 3), dtype=np.uint8)
        _, ratio, (pad_l, pad_t) = letterbox(img, 320)
        # forward-map an original point, then apply decode's unmap
        ox, oy = 50.0, 30.0
        lx, ly = ox * ratio + pad_l, oy * ratio + pad_t
        assert (lx - pad_l) / ratio == pytest.approx(ox)
        assert (ly - pad_t) / ratio == pytest.approx(oy)


# --------------------------------------------------------------------------- #
# NMS                                                                          #
# --------------------------------------------------------------------------- #
class TestNmsNumpy:
    def test_suppresses_overlap_keeps_highest(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=float)
        scores = np.array([0.9, 0.8, 0.7])
        keep = nms_numpy(boxes, scores, 0.5)
        assert keep[0] == 0
        assert 1 not in keep
        assert 2 in keep

    def test_disjoint_boxes_all_kept(self) -> None:
        boxes = np.array([[0, 0, 5, 5], [20, 20, 25, 25]], dtype=float)
        assert sorted(nms_numpy(boxes, np.array([0.5, 0.6]), 0.5)) == [0, 1]

    def test_empty(self) -> None:
        assert nms_numpy(np.zeros((0, 4)), np.zeros(0), 0.5) == []


# --------------------------------------------------------------------------- #
# keypoint_count                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("channels", "expected"), [(5, 0), (6, 0), (20, 5), (56, 17)])
def test_keypoint_count(channels: int, expected: int) -> None:
    assert keypoint_count(channels) == expected


# --------------------------------------------------------------------------- #
# decode_yolo_output                                                           #
# --------------------------------------------------------------------------- #
def _raw_det(anchors: list[tuple[float, float, float, float, float]]) -> np.ndarray:
    """Build a (1, 5, A) raw detection tensor from (cx, cy, w, h, score) rows."""
    return np.array(anchors, dtype=np.float32).T[None]


class TestDecodeYoloOutput:
    def test_single_box_identity_letterbox(self) -> None:
        raw = _raw_det([(100, 100, 40, 40, 0.9), (10, 10, 4, 4, 0.1)])
        dets = decode_yolo_output(
            raw, ratio=1.0, pad=(0, 0), orig_shape=(200, 200), conf=0.5, iou=0.5, n_kpt=0
        )
        assert len(dets) == 1
        d = dets[0]
        assert (d.x1, d.y1, d.x2, d.y2) == (80, 80, 120, 120)
        assert d.score == pytest.approx(0.9)
        assert d.landmarks is None

    def test_infers_orientation_without_n_kpt(self) -> None:
        # many anchors so the smaller-axis heuristic applies
        raw = _raw_det([(100, 100, 40, 40, 0.9)] + [(0, 0, 1, 1, 0.0)] * 20)
        dets = decode_yolo_output(
            raw, ratio=1.0, pad=(0, 0), orig_shape=(200, 200), conf=0.5, iou=0.5
        )
        assert len(dets) == 1 and dets[0].x2 == 120

    def test_letterbox_unmap_applied(self) -> None:
        raw = _raw_det([(100, 100, 40, 40, 0.95)])
        dets = decode_yolo_output(
            raw, ratio=0.5, pad=(10, 20), orig_shape=(1000, 1000), conf=0.5, iou=0.5, n_kpt=0
        )
        # x1_lb = 80 -> (80 - 10) / 0.5 = 140 ; y1_lb = 80 -> (80 - 20) / 0.5 = 120
        assert dets[0].x1 == pytest.approx(140)
        assert dets[0].y1 == pytest.approx(120)

    def test_confidence_filter(self) -> None:
        raw = _raw_det([(100, 100, 40, 40, 0.4), (50, 50, 20, 20, 0.49)])
        assert (
            decode_yolo_output(
                raw, ratio=1.0, pad=(0, 0), orig_shape=(200, 200), conf=0.5, iou=0.5, n_kpt=0
            )
            == []
        )

    def test_boxes_clipped_to_frame(self) -> None:
        raw = _raw_det([(10, 10, 60, 60, 0.9)])  # extends to -20,-20 .. 40,40
        dets = decode_yolo_output(
            raw, ratio=1.0, pad=(0, 0), orig_shape=(100, 100), conf=0.5, iou=0.5, n_kpt=0
        )
        assert dets[0].x1 == 0 and dets[0].y1 == 0
        assert dets[0].x2 == 40

    def test_pose_output_yields_landmarks(self) -> None:
        row = [120.0, 120.0, 50.0, 50.0, 0.9]
        for i in range(5):
            row += [100.0 + i, 110.0 + i, 0.9]  # x, y, visibility
        raw = np.array([row], dtype=np.float32).T[None]  # (1, 20, 1)
        dets = decode_yolo_output(
            raw, ratio=1.0, pad=(0, 0), orig_shape=(300, 300), conf=0.5, iou=0.5, n_kpt=5
        )
        assert len(dets) == 1
        assert dets[0].landmarks is not None
        assert dets[0].landmarks.shape == (5, 2)
        assert dets[0].landmarks[0].tolist() == [100.0, 110.0]


# --------------------------------------------------------------------------- #
# latency helpers                                                              #
# --------------------------------------------------------------------------- #
class TestMeasureLatency:
    def test_reflects_call_time(self) -> None:
        stats = measure_latency(lambda _: time.sleep(0.003), [None] * 8, warmup=2, label="x")
        assert stats.n == 8
        assert stats.mean_ms >= 2.0
        assert stats.p95_ms >= stats.p50_ms
        assert stats.fps == pytest.approx(1000.0 / stats.mean_ms, rel=1e-6)

    def test_render_report_has_both_rows_and_speedup(self) -> None:
        base = LatencyStats("PyTorch", 50, 20.0, 19.0, 30.0, 50.0)
        opt = LatencyStats("ONNX Runtime", 50, 8.0, 7.5, 12.0, 125.0)
        md = render_latency_report(base, opt, {"device": "cpu"})
        assert "PyTorch" in md and "ONNX Runtime" in md
        assert "1.00x" in md
        assert "2.50x" in md  # 20 / 8


# --------------------------------------------------------------------------- #
# registry wiring                                                              #
# --------------------------------------------------------------------------- #
def test_registry_onnx_runtime_builds_onnx_detector(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from app.detectors.registry import build_detector

    monkeypatch.setattr("app.detectors.registry.settings.detector_runtime", "onnx")
    monkeypatch.setattr("app.config.settings.onnx_model_path", tmp_path / "missing.onnx")
    # wiring reached OnnxFaceDetector (which fails only because the file is absent)
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        build_detector()


@pytest.fixture
def onnx_detector():
    from pathlib import Path

    from app.config import settings

    p = settings.onnx_model_path
    p = p if p.is_absolute() else (Path.cwd() / p)
    if not p.exists():
        pytest.skip(f"ONNX model absent ({settings.onnx_model_path})")
    from app.detectors.onnx_detector import OnnxFaceDetector

    return OnnxFaceDetector()


class TestOnnxDetectorIntegration:
    def test_blank_frame_no_faces(self, onnx_detector, blank_image: np.ndarray) -> None:
        assert onnx_detector.detect(blank_image, 0.5) == []

    def test_returns_detection_list(self, onnx_detector) -> None:
        out = onnx_detector.detect(np.zeros((128, 128, 3), dtype=np.uint8), 0.5)
        assert isinstance(out, list)
        assert all(isinstance(d, Detection) for d in out)
