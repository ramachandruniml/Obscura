"""Video redaction pipeline.

demux (PyAV) -> for each frame: detect every Nth frame + ByteTrack, coast the
rest -> redact every frame in place -> re-encode (libx264) -> mux the source
audio back with ffmpeg. Output is always .mp4 so the browser can preview it.

Frames are decoded one at a time and never held past the loop body.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from app.config import settings
from app.detectors.base import Detector
from app.errors import CorruptedMediaError
from app.logging import get_logger
from app.redaction.redactors import Redactor
from app.tracking import FaceTracker

log = get_logger(__name__)

_PROGRESS_EVERY = 100


@dataclass(frozen=True, slots=True)
class VideoResult:
    n_frames: int
    n_detection_frames: int
    n_tracks: int
    fps: float
    width: int
    height: int
    has_audio: bool


def _even(n: int) -> int:
    return n - (n % 2)


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return shutil.which("ffmpeg")


def _mux_audio(video_only: Path, source: Path, dst: Path) -> bool:
    exe = _ffmpeg_exe()
    if not exe:
        return False
    cmd = [
        exe, "-y",
        "-i", str(video_only),
        "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(dst),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        log.warning("video.audio_mux_failed", stderr=proc.stderr[-400:])
        return False
    return True


def redact_video(
    src: Path,
    dst: Path,
    detector: Detector,
    redactor: Redactor,
    confidence: float,
    *,
    detect_every_n: int | None = None,
) -> VideoResult:
    import av

    n = max(1, detect_every_n or settings.detect_every_n_frames)
    log.info("pipeline.video.start", src=src.name, dst=dst.name, detect_every_n=n)

    try:
        container = av.open(str(src))
    except Exception as exc:  # noqa: BLE001
        raise CorruptedMediaError(f"could not open video: {exc}") from exc

    idx = 0
    n_det = 0
    seen: set[int] = set()
    has_audio = False

    with container:
        if not container.streams.video:
            raise CorruptedMediaError("no video stream")
        in_v = container.streams.video[0]
        in_v.thread_type = "AUTO"
        has_audio = bool(container.streams.audio)

        rate = in_v.average_rate or Fraction(settings.track_fps_default, 1)
        fps = float(rate)
        width = _even(in_v.codec_context.width)
        height = _even(in_v.codec_context.height)
        if width < 2 or height < 2:
            raise CorruptedMediaError("video frame too small")

        tracker = FaceTracker(
            frame_size=(width, height),
            frame_rate=round(fps) or settings.track_fps_default,
            detect_every_n=n,
        )

        tmp = dst.with_name("_video_only.mp4")
        out_c = av.open(str(tmp), mode="w")
        out_v = out_c.add_stream(settings.video_output_codec, rate=rate)
        out_v.width, out_v.height = width, height
        out_v.pix_fmt = settings.video_output_pix_fmt
        out_v.options = {"crf": str(settings.video_output_crf)}

        try:
            for vframe in container.decode(video=0):
                img = np.ascontiguousarray(vframe.to_ndarray(format="bgr24")[:height, :width])

                if idx % n == 0:
                    detections = detector.detect(img, confidence)
                    n_det += 1
                    tracks = tracker.update(idx, detections)
                else:
                    tracks = tracker.update(idx, None)
                seen.update(t.track_id for t in tracks)

                redactor.apply(img, tracks, copy=False)

                for pkt in out_v.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                    out_c.mux(pkt)
                idx += 1
                if idx % _PROGRESS_EVERY == 0:
                    log.info("pipeline.video.progress", frames=idx, tracks=len(seen))

            for pkt in out_v.encode():
                out_c.mux(pkt)
        finally:
            out_c.close()

    if idx == 0:
        tmp.unlink(missing_ok=True)
        raise CorruptedMediaError("video contained no decodable frames")

    log.info("tracking.completed", n_tracks=len(seen), n_detection_frames=n_det, frames=idx)

    if has_audio and _mux_audio(tmp, src, dst):
        tmp.unlink(missing_ok=True)
    else:
        tmp.replace(dst)
        has_audio = False

    log.info("job.completed", kind="video", frames=idx, n_tracks=len(seen))
    return VideoResult(
        n_frames=idx,
        n_detection_frames=n_det,
        n_tracks=len(seen),
        fps=fps,
        width=width,
        height=height,
        has_audio=has_audio,
    )
