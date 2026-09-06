"""Upload classification, size-limited streaming to disk, and media probing.

Everything user-facing that can go wrong with an upload is raised as an
``ObscuraError`` subclass so the API can map it to a status code and the worker
never sees a malformed file.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Literal

from app.config import settings
from app.errors import (
    CorruptedMediaError,
    MediaTooLargeError,
    MediaTooLongError,
    UnsupportedMediaError,
)
from app.logging import get_logger

log = get_logger(__name__)

_EXT_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
}

Kind = Literal["image", "video"]


def classify(content_type: str | None, filename: str | None) -> tuple[Kind, str]:
    """Return ``(kind, extension)`` for an accepted upload, else raise UnsupportedMediaError."""
    mime = (content_type or "").split(";")[0].strip().lower()
    kind = settings.kind_for_mime(mime)
    if kind is None:
        raise UnsupportedMediaError(
            f"unsupported type {mime or '(none)'}; allowed: {', '.join(settings.allowed_types)}"
        )
    ext = _EXT_BY_MIME.get(mime)
    if ext is None and filename and "." in filename:
        ext = "." + filename.rsplit(".", 1)[1].lower()
    return kind, ext or (".jpg" if kind == "image" else ".mp4")


def stream_to_file(src: BinaryIO, dst: Path, *, max_bytes: int | None = None) -> int:
    """Copy ``src`` to ``dst`` in chunks, aborting past ``max_bytes``. Returns bytes written."""
    limit = settings.max_upload_bytes if max_bytes is None else max_bytes
    written = 0
    with dst.open("wb") as out:
        while chunk := src.read(1 << 20):
            written += len(chunk)
            if written > limit:
                out.close()
                dst.unlink(missing_ok=True)
                raise MediaTooLargeError(f"upload exceeds {limit} bytes (max_upload_bytes)")
            out.write(chunk)
    return written


def probe_image(path: Path) -> tuple[int, int]:
    """Validate an image file and return ``(width, height)``."""
    from PIL import Image

    prev_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
    try:
        with Image.open(path) as im:
            im.verify()  # catches truncation / bad headers
        with Image.open(path) as im:
            w, h = im.size
    except MediaTooLargeError:
        raise
    except Exception as exc:  # noqa: BLE001 - PIL raises a zoo of exception types
        raise CorruptedMediaError(f"could not decode image: {exc}") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = prev_limit

    if w * h > settings.max_image_pixels:
        raise CorruptedMediaError(
            f"image {w}x{h} exceeds max_image_pixels ({settings.max_image_pixels})"
        )
    return w, h


def probe_video(path: Path) -> tuple[int, int, float]:
    """Validate a video file and return ``(width, height, duration_seconds)``."""
    import av

    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise CorruptedMediaError("no video stream found")
            stream = container.streams.video[0]
            width, height = stream.codec_context.width, stream.codec_context.height
            duration = 0.0
            if container.duration is not None:
                duration = float(container.duration) / 1_000_000.0  # av.time_base
            elif stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
    except CorruptedMediaError:
        raise
    except Exception as exc:  # noqa: BLE001 - av raises av.AVError and friends
        raise CorruptedMediaError(f"could not open video: {exc}") from exc

    if not width or not height:
        raise CorruptedMediaError("video has no usable dimensions")
    if duration and duration > settings.max_video_duration_seconds:
        raise MediaTooLongError(
            f"video is {duration:.1f}s; max is {settings.max_video_duration_seconds}s"
        )
    return width, height, duration


def probe(path: Path, kind: Kind) -> dict[str, float]:
    if kind == "image":
        w, h = probe_image(path)
        return {"width": w, "height": h}
    w, h, dur = probe_video(path)
    return {"width": w, "height": h, "duration": dur}
