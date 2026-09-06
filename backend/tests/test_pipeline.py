"""Image + video pipelines, driven by a fake detector (no model weights)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from app.detectors.base import Detection
from app.errors import CorruptedMediaError
from app.pipeline.image_pipeline import redact_image
from app.pipeline.video_pipeline import redact_video
from app.redaction.redactors import SolidBoxRedactor

BOX = Detection(10, 8, 44, 40, 0.95)


def _write_image(path: Path, w: int = 96, h: int = 72) -> None:
    rng = np.random.default_rng(3)
    cv2.imwrite(str(path), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


class TestImagePipeline:
    def test_redacts_detected_region(self, tmp_path: Path, make_fake_detector) -> None:
        src, dst = tmp_path / "in.png", tmp_path / "out.png"
        _write_image(src)
        result = redact_image(src, dst, make_fake_detector([BOX]), SolidBoxRedactor(), 0.5)

        assert dst.exists()
        assert result.n_faces == 1
        out = cv2.imread(str(dst))
        x1, y1, x2, y2 = BOX.as_int_box()
        assert np.all(out[y1:y2, x1:x2] == 0)

    def test_zero_faces_still_writes_valid_output(self, tmp_path: Path, make_fake_detector) -> None:
        src, dst = tmp_path / "in.jpg", tmp_path / "out.jpg"
        _write_image(src)
        result = redact_image(src, dst, make_fake_detector([]), SolidBoxRedactor(), 0.5)
        assert result.n_faces == 0
        assert cv2.imread(str(dst)) is not None

    def test_corrupted_image_raises(self, tmp_path: Path, make_fake_detector) -> None:
        src, dst = tmp_path / "in.png", tmp_path / "out.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage")
        with pytest.raises(CorruptedMediaError):
            redact_image(src, dst, make_fake_detector([]), SolidBoxRedactor(), 0.5)

    def test_dimensions_reported(self, tmp_path: Path, make_fake_detector) -> None:
        src, dst = tmp_path / "in.png", tmp_path / "out.png"
        _write_image(src, w=120, h=90)
        result = redact_image(src, dst, make_fake_detector([]), SolidBoxRedactor(), 0.5)
        assert (result.width, result.height) == (120, 90)


class TestVideoPipeline:
    def test_processes_all_frames_and_redacts(
        self, tmp_path: Path, make_video, make_fake_detector
    ) -> None:
        src = make_video(tmp_path / "in.mp4", n_frames=15, w=64, h=48)
        dst = tmp_path / "out.mp4"
        result = redact_video(
            src, dst, make_fake_detector([BOX]), SolidBoxRedactor(), 0.5, detect_every_n=5
        )

        assert dst.exists()
        assert result.n_frames == 15
        assert result.n_detection_frames == 3  # frames 0, 5, 10
        assert result.n_tracks >= 1

        import av

        with av.open(str(dst)) as c:
            frames = list(c.decode(video=0))
        assert 13 <= len(frames) <= 15  # encoder may drop/pad a frame or two
        mid = frames[len(frames) // 2].to_ndarray(format="bgr24")
        x1, y1, x2, y2 = BOX.as_int_box()
        assert mid[y1:y2, x1:x2].mean() < 40  # region is (near) black

    def test_no_faces_video_is_untouched_content(
        self, tmp_path: Path, make_video, make_fake_detector
    ) -> None:
        src = make_video(tmp_path / "in.mp4", n_frames=8)
        dst = tmp_path / "out.mp4"
        result = redact_video(src, dst, make_fake_detector([]), SolidBoxRedactor(), 0.5)
        assert result.n_tracks == 0
        assert dst.exists()

    def test_video_with_audio_still_outputs(
        self, tmp_path: Path, make_video, make_fake_detector
    ) -> None:
        src = make_video(tmp_path / "in.mp4", n_frames=10, with_audio=True)
        dst = tmp_path / "out.mp4"
        result = redact_video(src, dst, make_fake_detector([BOX]), SolidBoxRedactor(), 0.5)
        assert dst.exists()
        assert result.n_frames == 10

    def test_empty_file_raises(self, tmp_path: Path, make_fake_detector) -> None:
        src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
        src.write_bytes(b"")
        with pytest.raises(CorruptedMediaError):
            redact_video(src, dst, make_fake_detector([]), SolidBoxRedactor(), 0.5)
