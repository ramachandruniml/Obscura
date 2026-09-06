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

from pathlib import Path

import cv2
import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
