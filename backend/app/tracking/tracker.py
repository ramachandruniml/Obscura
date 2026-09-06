"""Multi-face tracking across video frames (ByteTrack via `supervision`).

The pipeline runs the detector only every Nth frame. This module:

* on detection frames  -> ``update(idx, detections)`` assigns stable track ids
* on in-between frames  -> ``update(idx, None)`` extrapolates each live track's
  box (constant velocity, size held) so redaction does not flicker

A coasted track is dropped when its predicted box leaves the frame or when it
has coasted for more than ``max_coast_frames`` without the detector confirming
it again. A face that re-enters after a long absence gets a NEW id (no
re-identification — that is deliberately out of scope for this project).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.detectors.base import Detection, iou
from app.logging import get_logger

log = get_logger(__name__)

_LANDMARK_MATCH_IOU = 0.5


@dataclass(frozen=True, slots=True)
class TrackedFace:
    """A face with a persistent id for the current frame.

    ``box`` is a plain :class:`Detection` (all geometry helpers apply).
    ``coasting`` is True when the box was extrapolated rather than detected;
    redaction should pad those boxes more generously.
    """

    track_id: int
    box: Detection
    coasting: bool = False
    age: int = 1  # updates (real or coast) since the track first appeared
    coast_age: int = 0  # frames since the last real detection for this track

    @property
    def score(self) -> float:
        return self.box.score

    @property
    def landmarks(self) -> np.ndarray | None:
        return self.box.landmarks


@dataclass
class _TrackState:
    track_id: int
    box: Detection  # last detected (real) box
    last_real_idx: int
    prev_real_idx: int | None = None
    prev_center: tuple[float, float] | None = None
    age: int = 1

    def velocity(self) -> tuple[float, float]:
        if self.prev_center is None or self.prev_real_idx is None:
            return 0.0, 0.0
        span = self.last_real_idx - self.prev_real_idx
        if span <= 0:
            return 0.0, 0.0
        cx, cy = self.box.center
        pcx, pcy = self.prev_center
        return (cx - pcx) / span, (cy - pcy) / span

    def extrapolate(self, frame_idx: int) -> Detection:
        vx, vy = self.velocity()
        dt = frame_idx - self.last_real_idx
        cx, cy = self.box.center
        ncx, ncy = cx + vx * dt, cy + vy * dt
        w, h = self.box.width, self.box.height
        return Detection(ncx - w / 2, ncy - h / 2, ncx + w / 2, ncy + h / 2, self.box.score)


def _build_bytetrack(
    frame_rate: int,
    activation_threshold: float,
    matching_threshold: float,
    lost_buffer: int,
):
    """Construct supervision's ByteTrack across the supported version range (>=0.25,<0.28).

    The constructor kwargs were renamed over supervision's history, so try the
    modern names first and fall back to the legacy ones.
    """
    from supervision import ByteTrack

    modern = {
        "track_activation_threshold": activation_threshold,
        "lost_track_buffer": lost_buffer,
        "minimum_matching_threshold": matching_threshold,
        "frame_rate": frame_rate,
    }
    legacy = {
        "track_thresh": activation_threshold,
        "track_buffer": lost_buffer,
        "match_thresh": matching_threshold,
        "frame_rate": frame_rate,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        for kwargs in (modern, legacy):
            try:
                return ByteTrack(**kwargs)
            except TypeError:
                continue
        return ByteTrack()


def _to_sv(detections: list[Detection]):
    import supervision as sv

    if not detections:
        return sv.Detections.empty()
    xyxy = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
    conf = np.array([d.score for d in detections], dtype=np.float32)
    class_id = np.zeros(len(detections), dtype=int)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)


def _match_landmarks(detections: list[Detection], box: Detection):
    best = None
    best_iou = _LANDMARK_MATCH_IOU
    for d in detections:
        v = iou(d, box)
        if v > best_iou:
            best_iou, best = v, d
    return best.landmarks if best is not None else None


class FaceTracker:
    """Stateful per-video tracker. Create one per job; call :meth:`update` per frame."""

    def __init__(
        self,
        *,
        frame_size: tuple[int, int],
        frame_rate: int | None = None,
        detect_every_n: int | None = None,
        activation_threshold: float | None = None,
        matching_threshold: float | None = None,
        lost_buffer: int | None = None,
        max_coast_frames: int | None = None,
    ) -> None:
        self._frame_w, self._frame_h = frame_size
        self._detect_every_n = detect_every_n or settings.detect_every_n_frames

        configured = (
            settings.track_max_coast_frames if max_coast_frames is None else max_coast_frames
        )
        self._max_coast_frames = configured if configured > 0 else 3 * self._detect_every_n

        self._bt = _build_bytetrack(
            frame_rate or settings.track_fps_default,
            activation_threshold
            if activation_threshold is not None
            else settings.track_activation_threshold,
            matching_threshold
            if matching_threshold is not None
            else settings.min_matching_threshold,
            lost_buffer if lost_buffer is not None else settings.track_buffer_frames,
        )
        self._states: dict[int, _TrackState] = {}

    def __len__(self) -> int:
        return len(self._states)

    @property
    def active_track_ids(self) -> list[int]:
        return sorted(self._states)

    def reset(self) -> None:
        self._bt.reset()
        self._states.clear()

    def update(self, frame_idx: int, detections: list[Detection] | None) -> list[TrackedFace]:
        """Advance one frame.

        ``detections=None`` -> coast every live track (no detector ran).
        ``detections=[]``   -> detector ran and found nothing.
        """
        matched: dict[int, Detection] = {}
        if detections is not None:
            tracked = self._bt.update_with_detections(_to_sv(detections))
            for i in range(len(tracked)):
                tid = tracked.tracker_id[i]
                if tid is None:
                    continue
                tid = int(tid)
                x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
                score = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                lm = _match_landmarks(detections, Detection(x1, y1, x2, y2, 0.0))
                box = Detection(x1, y1, x2, y2, score, lm).clip(self._frame_w, self._frame_h)
                if box.is_degenerate():
                    continue
                matched[tid] = box
                self._absorb(tid, box, frame_idx)

        return self._emit(frame_idx, matched, is_detection_frame=detections is not None)

    # -- internals ---------------------------------------------------------------
    def _absorb(self, tid: int, box: Detection, frame_idx: int) -> None:
        st = self._states.get(tid)
        if st is None:
            self._states[tid] = _TrackState(tid, box, frame_idx, age=1)
            return
        if frame_idx != st.last_real_idx:
            st.prev_real_idx = st.last_real_idx
            st.prev_center = st.box.center
        st.box = box
        st.last_real_idx = frame_idx
        st.age += 1

    def _emit(
        self, frame_idx: int, matched: dict[int, Detection], *, is_detection_frame: bool
    ) -> list[TrackedFace]:
        out: list[TrackedFace] = []
        dropped: list[int] = []

        for tid, st in self._states.items():
            if tid in matched:
                out.append(TrackedFace(tid, matched[tid], coasting=False, age=st.age, coast_age=0))
                continue

            coast_age = frame_idx - st.last_real_idx
            if coast_age <= 0:
                continue  # already emitted as matched this frame
            if coast_age > self._max_coast_frames:
                dropped.append(tid)
                continue

            predicted = st.extrapolate(frame_idx).clip(self._frame_w, self._frame_h)
            if predicted.is_degenerate():
                dropped.append(tid)  # track has left the frame
                continue

            st.age += 1
            out.append(TrackedFace(tid, predicted, coasting=True, age=st.age, coast_age=coast_age))

        for tid in dropped:
            self._states.pop(tid, None)

        log.debug(
            "tracking.update",
            frame=frame_idx,
            detection_frame=is_detection_frame,
            matched=len(matched),
            coasting=sum(1 for t in out if t.coasting),
            active=len(self._states),
        )
        return out
