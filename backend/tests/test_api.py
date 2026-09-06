"""API: POST /redact, GET /jobs/{id}, GET /jobs/{id}/result.

Runs the pipeline inline via eager Celery + a fake detector — no broker, no
model weights.
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np
import pytest
from app import storage
from app.config import settings
from app.detectors.base import Detection
from app.main import app
from fastapi.testclient import TestClient

BOX = Detection(12, 10, 60, 58, 0.95)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def _png(w: int = 96, h: int = 80) -> bytes:
    rng = np.random.default_rng(1)
    ok, buf = cv2.imencode(".png", rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _post(client: TestClient, *, data: bytes, name: str, mime: str, **form: object):
    return client.post(
        "/redact",
        files={"file": (name, data, mime)},
        data={k: str(v) for k, v in form.items()},
    )


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
class TestValidation:
    def test_unsupported_type_is_415(self, client: TestClient) -> None:
        r = _post(client, data=b"hello", name="notes.txt", mime="text/plain")
        assert r.status_code == 415

    def test_unknown_method_is_400(self, client: TestClient) -> None:
        r = _post(client, data=_png(), name="a.png", mime="image/png", method="scramble")
        assert r.status_code == 400

    def test_out_of_range_confidence_is_400(self, client: TestClient) -> None:
        r = _post(client, data=_png(), name="a.png", mime="image/png", confidence=5)
        assert r.status_code == 400

    def test_corrupted_image_is_422(self, client: TestClient) -> None:
        r = _post(client, data=b"\x89PNG\r\n\x1a\nnope", name="a.png", mime="image/png")
        assert r.status_code == 422

    def test_oversized_upload_is_413(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "max_upload_bytes", 16)
        r = _post(client, data=_png(), name="a.png", mime="image/png")
        assert r.status_code == 413

    def test_rejected_upload_leaves_no_job_dir(self, client: TestClient) -> None:
        _post(client, data=b"\x89PNG\r\n\x1a\nnope", name="a.png", mime="image/png")
        assert list(settings.storage_dir.glob("*")) == []


# --------------------------------------------------------------------------- #
# Happy path                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("eager_celery")
class TestImageJob:
    def test_end_to_end(self, client: TestClient, patch_detector, make_fake_detector) -> None:
        patch_detector(make_fake_detector([BOX]))

        r = _post(
            client, data=_png(), name="me.png", mime="image/png", method="box", confidence=0.5
        )
        assert r.status_code == 202
        job_id = r.json()["job_id"]

        job = client.get(f"/jobs/{job_id}").json()
        assert job["status"] == "complete"
        assert job["stats"]["n_faces"] == 1
        assert job["result_url"] == f"/jobs/{job_id}/result"

        res = client.get(job["result_url"])
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/png"
        out = cv2.imdecode(np.frombuffer(res.content, np.uint8), cv2.IMREAD_COLOR)
        x1, y1, x2, y2 = BOX.as_int_box()
        assert np.all(out[y1:y2, x1:x2] == 0)

    def test_input_deleted_after_success(
        self, client: TestClient, patch_detector, make_fake_detector
    ) -> None:
        patch_detector(make_fake_detector([BOX]))
        job_id = _post(client, data=_png(), name="a.png", mime="image/png").json()["job_id"]
        assert storage.input_path(job_id) is None
        assert storage.result_path(job_id) is not None

    def test_zero_faces_completes(
        self, client: TestClient, patch_detector, make_fake_detector
    ) -> None:
        patch_detector(make_fake_detector([]))
        job_id = _post(client, data=_png(), name="a.png", mime="image/png").json()["job_id"]
        job = client.get(f"/jobs/{job_id}").json()
        assert job["status"] == "complete"
        assert job["stats"]["n_faces"] == 0

    def test_pipeline_failure_marks_job_failed(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # a worker crash must not take down the request: POST still returns 202,
        # and the job manifest records the failure for the client to poll.
        from app.worker.celery_app import celery

        monkeypatch.setattr(celery.conf, "task_eager_propagates", False)

        def boom() -> None:
            raise RuntimeError("model exploded")

        monkeypatch.setattr("app.pipeline.runner.get_cached_detector", boom)
        r = _post(client, data=_png(), name="a.png", mime="image/png")
        assert r.status_code == 202
        job = client.get(f"/jobs/{r.json()['job_id']}").json()
        assert job["status"] == "failed"
        assert "model exploded" in job["error"]


@pytest.mark.usefixtures("eager_celery")
class TestVideoJob:
    def test_end_to_end(
        self, client: TestClient, patch_detector, make_fake_detector, make_video, tmp_path
    ) -> None:
        patch_detector(make_fake_detector([BOX]))
        src = make_video(tmp_path / "clip.mp4", n_frames=12, w=96, h=64)

        r = _post(
            client, data=src.read_bytes(), name="clip.mp4", mime="video/mp4", method="pixelate"
        )
        assert r.status_code == 202
        job_id = r.json()["job_id"]

        job = client.get(f"/jobs/{job_id}").json()
        assert job["status"] == "complete"
        assert job["stats"]["n_frames"] == 12

        res = client.get(job["result_url"])
        assert res.status_code == 200
        assert res.headers["content-type"] == "video/mp4"
        assert len(res.content) > 0


# --------------------------------------------------------------------------- #
# Job lookup edge cases                                                        #
# --------------------------------------------------------------------------- #
class TestJobLookup:
    def test_unknown_job_is_404(self, client: TestClient) -> None:
        assert client.get("/jobs/deadbeef").status_code == 404
        assert client.get("/jobs/deadbeef/result").status_code == 404

    def test_result_before_complete_is_404(self, client: TestClient) -> None:
        job_id = storage.new_job_id()
        storage.create_job_dir(job_id)
        storage.init_manifest(
            job_id, kind="image", method="blur", confidence=0.5, input_name="x.png"
        )
        storage.update_manifest(job_id, status="processing")
        assert client.get(f"/jobs/{job_id}/result").status_code == 404
        assert client.get(f"/jobs/{job_id}").json()["result_url"] is None
