"""Domain errors shared by the media/validation layer, pipelines, and the API.

The API layer maps these to HTTP status codes (see app/main.py exception
handlers); the worker records ``str(exc)`` on the job manifest.
"""

from __future__ import annotations


class ObscuraError(Exception):
    """Base class for all expected (non-bug) failures."""

    http_status = 400


class UnsupportedMediaError(ObscuraError):
    """File type / MIME is not in the configured allow-list."""

    http_status = 415


class MediaTooLargeError(ObscuraError):
    """Upload exceeds ``settings.max_upload_bytes``."""

    http_status = 413


class MediaTooLongError(ObscuraError):
    """Video duration exceeds ``settings.max_video_duration_seconds``."""

    http_status = 422


class CorruptedMediaError(ObscuraError):
    """File could not be decoded as a valid image / video."""

    http_status = 422


class JobNotFoundError(ObscuraError):
    """No job manifest for the given id (never created, or already swept)."""

    http_status = 404
