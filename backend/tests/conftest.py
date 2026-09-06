"""Shared test fixtures.

Populated alongside each module:
    Deliverable 2 - image fixtures + weight-gated detector fixtures
    Deliverable 3 - synthetic_track_sequence (faces entering/leaving frame)
    Deliverable 4 - sample_boxes, frame factory
    Deliverable 5 - api client, fake celery (eager mode), tmp storage_dir

Run pytest from the `backend/` directory so relative weight paths resolve.
Integration tests that need model weights or a real face image skip themselves
when those files are absent (see `_weights_present` / `face_image`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest
from app.config import settings
from app.detectors.base import Detection, Detector

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------- #
# D5: storage isolation, eager celery, fake detector, video factory            #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def isolate_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point settings.storage_dir at a per-test temp dir (autouse: harmless elsewhere)."""
    d = tmp_path / "obscura-data"
    monkeypatch.setattr(settings, "storage_dir", d)
    return d


@pytest.fixture
def eager_celery(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.worker.celery_app import celery

    monkeypatch.setattr(celery.conf, "task_always_eager", True)
    monkeypatch.setattr(celery.conf, "task_eager_propagates", True)


class _FakeDetector(Detector):
    """Returns a fixed set of detections, filtered by the confidence contract."""

    name = "fake"
    provides_landmarks = False

    def __init__(self, boxes: Sequence[Detection]) -> None:
        super().__init__()
        self._boxes = list(boxes)

    def _detect(self, frame: np.ndarray, confidence: float) -> list[Detection]:
        return list(self._boxes)


@pytest.fixture
def make_fake_detector() -> Callable[[Sequence[Detection]], _FakeDetector]:
    return _FakeDetector


@pytest.fixture
def patch_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Detector], None]:
    def _patch(detector: Detector) -> None:
        monkeypatch.setattr("app.pipeline.runner.get_cached_detector", lambda: detector)

    return _patch


@pytest.fixture
def make_video() -> Callable[..., Path]:
    def _make(
        path: Path,
        *,
        n_frames: int = 12,
        w: int = 64,
        h: int = 48,
        fps: int = 10,
        with_audio: bool = False,
    ) -> Path:
        import av

        container = av.open(str(path), mode="w")
        vstream = container.add_stream("libx264", rate=fps)
        vstream.width, vstream.height = w, h
        vstream.pix_fmt = "yuv420p"

        astream = None
        if with_audio:
            astream = container.add_stream("aac", rate=44100)

        rng = np.random.default_rng(7)
        for i in range(n_frames):
            arr = np.full((h, w, 3), (i * 7) % 256, dtype=np.uint8)
            arr[:, :, 1] = rng.integers(0, 255)
            frame = av.VideoFrame.from_ndarray(arr, format="bgr24")
            for pkt in vstream.encode(frame):
                container.mux(pkt)
        for pkt in vstream.encode():
            container.mux(pkt)

        if astream is not None:
            samples = np.zeros((1, 44100 * n_frames // fps), dtype=np.int16)
            aframe = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
            aframe.sample_rate = 44100
            for pkt in astream.encode(aframe):
                container.mux(pkt)
            for pkt in astream.encode():
                container.mux(pkt)

        container.close()
        return path

    return _make


# --------------------------------------------------------------------------- #
# Image fixtures                                                               #
# --------------------------------------------------------------------------- #
@pytest.fixture
def blank_image() -> np.ndarray:
    """640x480 mid-grey BGR frame with no faces."""
    return np.full((480, 640, 3), 127, dtype=np.uint8)


@pytest.fixture
def noise_image(rng: np.random.Generator) -> np.ndarray:
    """Random RGB noise — no face structure."""
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_face_image() -> np.ndarray:
    """A crude drawn 'face' (ellipse + eyes + mouth).

    Enough for redaction/geometry tests. Real detectors are NOT expected to
    fire on this — detection integration tests use `face_image` instead.
    """
    img = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.ellipse(img, (320, 240), (90, 120), 0, 0, 360, (170, 150, 140), -1)
    cv2.circle(img, (290, 210), 12, (40, 40, 40), -1)
    cv2.circle(img, (350, 210), 12, (40, 40, 40), -1)
    cv2.ellipse(img, (320, 285), (35, 18), 0, 0, 180, (60, 60, 60), 4)
    return img


@pytest.fixture
def corrupted_image_bytes() -> bytes:
    """JPEG magic bytes followed by garbage — decodes to None via cv2/PIL."""
    return b"\xff\xd8\xff\xe0" + b"\x00" * 32 + b"not really a jpeg"


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture(scope="session")
def face_image() -> np.ndarray:
    """A real face image for detection integration tests.

    Drop any image containing at least one clear face at
    `backend/tests/fixtures/faces.jpg` (png/jpeg). Tests skip if it is missing.
    """
    for name in ("faces.jpg", "faces.jpeg", "faces.png"):
        path = FIXTURES_DIR / name
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                return img
    pytest.skip("no tests/fixtures/faces.{jpg,png} — add one to run detection integration tests")


# --------------------------------------------------------------------------- #
# Weight-gated detector fixtures                                               #
# --------------------------------------------------------------------------- #
def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def _weights_present(path: Path) -> bool:
    return _resolve(path).exists()


@pytest.fixture(scope="session")
def retinaface_detector():
    from app.config import settings

    if not _weights_present(settings.retinaface_weights):
        pytest.skip(f"RetinaFace weights absent ({settings.retinaface_weights})")
    from app.detectors.retinaface import RetinaFaceDetector

    return RetinaFaceDetector()


@pytest.fixture(scope="session")
def yolov8_detector():
    from app.config import settings

    if not _weights_present(settings.yolov8_face_weights):
        pytest.skip(f"YOLOv8-face weights absent ({settings.yolov8_face_weights})")
    from app.detectors.yolov8_face import YOLOv8FaceDetector

    return YOLOv8FaceDetector()


@pytest.fixture(params=["retinaface", "yolov8face"])
def any_detector(request: pytest.FixtureRequest):
    """Parametrized over both backends; each param skips if its weights are absent."""
    return request.getfixturevalue(
        {"retinaface": "retinaface_detector", "yolov8face": "yolov8_detector"}[request.param]
    )
