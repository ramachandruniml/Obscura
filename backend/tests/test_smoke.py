"""Skeleton smoke tests — replaced/expanded per deliverable."""

from __future__ import annotations

from app.config import settings
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_healthz() -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz_reports_backend() -> None:
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["detector_backend"] in {"retinaface", "yolov8face"}


def test_settings_csv_parsing() -> None:
    assert "image/jpeg" in settings.allowed_image_types
    assert settings.kind_for_mime("video/mp4") == "video"
    assert settings.kind_for_mime("application/zip") is None
