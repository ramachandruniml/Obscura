"""Local job storage + retention sweep."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest
from app import storage
from app.config import settings
from app.errors import JobNotFoundError


def test_create_and_read_manifest() -> None:
    job_id = storage.new_job_id()
    storage.create_job_dir(job_id)
    storage.init_manifest(job_id, kind="image", method="blur", confidence=0.5, input_name="x.jpg")

    m = storage.read_manifest(job_id)
    assert m is not None
    assert m["job_id"] == job_id
    assert m["status"] == "pending"
    assert m["kind"] == "image"
    assert datetime.fromisoformat(m["expires_at"]) > datetime.fromisoformat(m["created_at"])


def test_read_manifest_missing_returns_none() -> None:
    assert storage.read_manifest("does-not-exist") is None


def test_read_manifest_corrupt_returns_none() -> None:
    job_id = storage.new_job_id()
    storage.create_job_dir(job_id)
    (storage.job_dir(job_id) / storage.MANIFEST_NAME).write_text("{ not json")
    assert storage.read_manifest(job_id) is None


def test_update_manifest_merges() -> None:
    job_id = storage.new_job_id()
    storage.create_job_dir(job_id)
    storage.init_manifest(job_id, kind="image", method="box", confidence=0.4, input_name="a.png")
    storage.update_manifest(job_id, status="complete", stats={"n_faces": 3})

    m = storage.read_manifest(job_id)
    assert m["status"] == "complete"
    assert m["stats"] == {"n_faces": 3}
    assert m["method"] == "box"  # untouched


def test_update_missing_manifest_raises() -> None:
    with pytest.raises(JobNotFoundError):
        storage.update_manifest("nope", status="complete")


def test_input_and_result_paths() -> None:
    job_id = storage.new_job_id()
    storage.create_job_dir(job_id)
    storage.init_manifest(job_id, kind="image", method="blur", confidence=0.5, input_name="i.jpg")
    (storage.job_dir(job_id) / "input.jpg").write_bytes(b"data")

    assert storage.input_path(job_id).name == "input.jpg"
    assert storage.result_path(job_id) is None  # no output_name yet

    (storage.job_dir(job_id) / "output.jpg").write_bytes(b"out")
    storage.update_manifest(job_id, output_name="output.jpg")
    assert storage.result_path(job_id).name == "output.jpg"


def test_delete_input() -> None:
    job_id = storage.new_job_id()
    storage.create_job_dir(job_id)
    (storage.job_dir(job_id) / "input.mp4").write_bytes(b"x")
    storage.delete_input(job_id)
    assert storage.input_path(job_id) is None


def test_purge_job() -> None:
    job_id = storage.new_job_id()
    storage.create_job_dir(job_id)
    storage.purge_job(job_id)
    assert not storage.job_dir(job_id).exists()


class TestSweep:
    def _job(self, *, expires_in: float) -> str:
        job_id = storage.new_job_id()
        storage.create_job_dir(job_id)
        now = datetime.now(UTC)
        storage.write_manifest(
            job_id,
            {
                "job_id": job_id,
                "status": "complete",
                "method": "blur",
                "confidence": 0.5,
                "created_at": now.isoformat(),
                "expires_at": (now + timedelta(seconds=expires_in)).isoformat(),
            },
        )
        return job_id

    def test_removes_expired_keeps_fresh(self) -> None:
        stale = self._job(expires_in=-10)
        fresh = self._job(expires_in=3600)

        removed = storage.sweep()
        assert removed == 1
        assert not storage.job_dir(stale).exists()
        assert storage.job_dir(fresh).exists()

    def test_no_manifest_dir_swept_by_age(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "result_ttl_seconds", 1)
        job_id = storage.new_job_id()
        d = storage.create_job_dir(job_id)
        old = time.time() - 60
        import os

        os.utime(d, (old, old))
        assert storage.sweep() == 1
        assert not d.exists()

    def test_sweep_on_empty_root_is_noop(self) -> None:
        assert storage.sweep() == 0
