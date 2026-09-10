"""Settings parsing — especially env values that pydantic-settings would
otherwise try (and fail) to JSON-decode.

Regression: DETECTOR / CORS / SOLID_BOX_COLOR env vars are plain CSV, not JSON.
Without `NoDecode` on the list/tuple fields, constructing Settings from a real
.env raised `SettingsError: error parsing value for field "cors_origins"`.
"""

from __future__ import annotations

import pytest
from app.config import Settings


def _settings(env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return Settings(_env_file=None)  # ignore any real .env, use only the env vars


class TestCsvEnvValues:
    def test_cors_origins_from_csv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _settings({"CORS_ORIGINS": "http://a.test,http://b.test:3000"}, monkeypatch)
        assert s.cors_origins == ["http://a.test", "http://b.test:3000"]

    def test_allowed_types_from_csv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _settings(
            {
                "ALLOWED_IMAGE_TYPES": "image/jpeg, image/png",
                "ALLOWED_VIDEO_TYPES": "video/mp4",
            },
            monkeypatch,
        )
        assert s.allowed_image_types == ["image/jpeg", "image/png"]  # whitespace trimmed
        assert s.allowed_video_types == ["video/mp4"]
        assert s.kind_for_mime("video/mp4") == "video"
        assert s.kind_for_mime("image/gif") is None

    def test_solid_box_color_from_csv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _settings({"SOLID_BOX_COLOR": "10,20,30"}, monkeypatch)
        assert s.solid_box_color == (10, 20, 30)
        assert isinstance(s.solid_box_color, tuple)

    def test_defaults_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for k in ("CORS_ORIGINS", "ALLOWED_IMAGE_TYPES", "ALLOWED_VIDEO_TYPES", "SOLID_BOX_COLOR"):
            monkeypatch.delenv(k, raising=False)
        s = Settings(_env_file=None)
        assert "image/jpeg" in s.allowed_image_types
        assert s.solid_box_color == (0, 0, 0)
        assert s.cors_origins  # non-empty default

    def test_full_env_file_shape_loads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # mirrors .env.example — this is exactly what broke the containers
        s = _settings(
            {
                "CORS_ORIGINS": "http://localhost:5173,http://localhost:3000",
                "ALLOWED_IMAGE_TYPES": "image/jpeg,image/png,image/webp",
                "ALLOWED_VIDEO_TYPES": "video/mp4,video/quicktime,video/webm",
                "SOLID_BOX_COLOR": "0,0,0",
                "DETECTOR_BACKEND": "retinaface",
            },
            monkeypatch,
        )
        assert s.detector_backend == "retinaface"
        assert len(s.allowed_types) == 6
