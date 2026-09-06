"""FaceTracker (ByteTrack + coasting).

No model weights needed — everything is driven by synthetic Detection
sequences. Covers: stable ids across frames, faces entering / leaving /
re-entering, coasting on non-detection frames, velocity extrapolation,
empty-detection frames, determinism, reset, and frame-bound clamping.
"""

from __future__ import annotations

import numpy as np
import pytest
from app.detectors.base import Detection
from app.tracking import FaceTracker, TrackedFace

FRAME = (640, 480)


def face(cx: float, cy: float, w: float = 40, h: float = 52, score: float = 0.9) -> Detection:
    return Detection(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score)


def linear(idx: int, x0: float, y0: float, vx: float, vy: float, **kw: float) -> Detection:
    return face(x0 + vx * idx, y0 + vy * idx, **kw)


@pytest.fixture
def tracker() -> FaceTracker:
    return FaceTracker(frame_size=FRAME)


# --------------------------------------------------------------------------- #
# Stable identity                                                              #
# --------------------------------------------------------------------------- #
class TestStableIdentity:
    def test_two_faces_keep_ids_across_frames(self, tracker: FaceTracker) -> None:
        id_sets = []
        for i in range(20):
            dets = [linear(i, 100, 240, 6, 0), linear(i, 520, 120, -4, 3)]
            out = tracker.update(i, dets)
            id_sets.append(tuple(sorted(t.track_id for t in out)))

        settled = id_sets[4:]
        assert len(settled[0]) == 2
        assert all(s == settled[0] for s in settled), id_sets

    def test_id_stable_across_detect_every_n_cadence(self, tracker: FaceTracker) -> None:
        """Realistic pipeline pattern: detect every 5th frame, coast the rest.

        A moderate-speed face (3 px/frame -> 15 px between detections) keeps one
        id for the whole clip.
        """
        n = 5
        seen_ids: set[int] = set()
        for i in range(40):
            dets = [linear(i, 80, 240, 3, 0)] if i % n == 0 else None
            for tf in tracker.update(i, dets):
                seen_ids.add(tf.track_id)
        assert seen_ids == {1}

    def test_ids_are_distinct_per_face(self, tracker: FaceTracker) -> None:
        for i in range(6):
            tracker.update(i, [linear(i, 100, 240, 5, 0), linear(i, 400, 300, 0, 0)])
        out = tracker.update(6, [linear(6, 100, 240, 5, 0), linear(6, 400, 300, 0, 0)])
        assert len({t.track_id for t in out}) == 2

    def test_all_outputs_are_tracked_faces(self, tracker: FaceTracker) -> None:
        out = tracker.update(0, [face(100, 100)])
        assert all(isinstance(t, TrackedFace) for t in out)


# --------------------------------------------------------------------------- #
# Faces entering / leaving / re-entering                                       #
# --------------------------------------------------------------------------- #
class TestEnterLeave:
    def test_new_face_entering_gets_new_id(self, tracker: FaceTracker) -> None:
        for i in range(5):
            tracker.update(i, [linear(i, 100, 240, 3, 0)])
        first_ids = {t.track_id for t in tracker.update(5, [linear(5, 100, 240, 3, 0)])}

        tracker.update(6, [linear(6, 100, 240, 3, 0), face(320, 300)])
        out = tracker.update(7, [linear(7, 100, 240, 3, 0), face(320, 300)])
        ids = {t.track_id for t in out}
        assert len(ids) == 2
        assert first_ids < ids  # original id retained, one new id added

    def test_face_leaving_frame_is_dropped(self) -> None:
        tr = FaceTracker(frame_size=(200, 200), max_coast_frames=999)
        for i in range(80):
            cx = 40 + i * 6
            dets = [face(cx, 100, w=36, h=44)] if cx < 220 else []
            tr.update(i, dets)
        assert tr.update(80, []) == []
        assert len(tr) == 0

    def test_reentering_face_is_tracked_again(self, tracker: FaceTracker) -> None:
        for i in range(5):
            tracker.update(i, [face(150, 240)])
        # gone long enough to exceed the lost buffer
        for i in range(5, 90):
            tracker.update(i, [])
        out = tracker.update(90, [face(150, 240)])
        # may take a couple of frames to re-activate
        if not out:
            out = tracker.update(91, [face(150, 240)])
        assert len(out) == 1


# --------------------------------------------------------------------------- #
# Coasting on non-detection frames                                             #
# --------------------------------------------------------------------------- #
class TestCoasting:
    def test_coast_frame_yields_extrapolated_box(self, tracker: FaceTracker) -> None:
        # three consecutive detections establish a clean 6 px/frame velocity
        for i in range(3):
            tracker.update(i, [linear(i, 100, 240, 6, 0)])
        out = tracker.update(3, None)
        assert len(out) == 1
        tf = out[0]
        assert tf.coasting is True
        assert tf.coast_age == 1
        expected_cx = 100 + 6 * 2 + 6  # center at f2 plus one frame of velocity
        assert tf.box.center[0] == pytest.approx(expected_cx, abs=1.0)

    def test_extrapolation_continues_over_multiple_coast_frames(self, tracker: FaceTracker) -> None:
        for i in range(3):
            tracker.update(i, [linear(i, 200, 200, 4, 2)])
        for dt in range(1, 5):
            (tf,) = tracker.update(2 + dt, None)
            assert tf.coast_age == dt
            assert tf.box.center[0] == pytest.approx(200 + 4 * 2 + 4 * dt, abs=1.5)
            assert tf.box.center[1] == pytest.approx(200 + 2 * 2 + 2 * dt, abs=1.5)

    def test_single_observation_coasts_by_holding_position(self, tracker: FaceTracker) -> None:
        tracker.update(0, [face(300, 200)])
        out = tracker.update(1, None)
        assert len(out) == 1
        assert out[0].box.center == pytest.approx((300.0, 200.0))

    def test_detection_frames_backfill_missed_tracks(self, tracker: FaceTracker) -> None:
        for i in range(4):
            tracker.update(i, [face(200, 200), face(400, 200)])
        # one face missing on this detection frame -> still emitted, coasting
        out = tracker.update(4, [face(200, 200)])
        assert len(out) == 2
        assert sum(t.coasting for t in out) == 1

    def test_coast_is_bounded_and_dropped(self, tracker: FaceTracker) -> None:
        tracker.update(0, [face(320, 240)])
        tracker.update(3, [face(320, 240)])
        last = 0
        for i in range(4, 60):
            out = tracker.update(i, None)
            last = len(out)
            if last == 0:
                break
        assert last == 0
        assert len(tracker) == 0

    def test_coasted_box_stays_in_frame(self) -> None:
        tr = FaceTracker(frame_size=(320, 240))
        tr.update(0, [face(250, 120, w=40, h=40)])
        tr.update(1, [face(270, 120, w=40, h=40)])  # heading toward the right edge
        for i in range(2, 40):
            for tf in tr.update(i, None):
                assert 0 <= tf.box.x1 <= tf.box.x2 <= 320
                assert 0 <= tf.box.y1 <= tf.box.y2 <= 240


# --------------------------------------------------------------------------- #
# Degenerate inputs                                                            #
# --------------------------------------------------------------------------- #
class TestEdgeInputs:
    def test_empty_detection_frame_from_start(self, tracker: FaceTracker) -> None:
        assert tracker.update(0, []) == []

    def test_none_before_any_detection(self, tracker: FaceTracker) -> None:
        assert tracker.update(0, None) == []

    def test_sustained_empty_frames_clear_all_tracks(self, tracker: FaceTracker) -> None:
        for i in range(3):
            tracker.update(i, [face(100, 100), face(300, 300)])
        for i in range(3, 80):
            tracker.update(i, [])
        assert tracker.update(80, []) == []
        assert len(tracker) == 0


# --------------------------------------------------------------------------- #
# Determinism / lifecycle                                                      #
# --------------------------------------------------------------------------- #
class TestLifecycle:
    def _run(self) -> list[tuple[tuple[int, int], ...]]:
        tr = FaceTracker(frame_size=FRAME)
        seq: list[tuple[tuple[int, int], ...]] = []
        for i in range(15):
            dets: list[Detection] | None
            dets = None if i % 3 else [linear(i, 120, 240, 4, 1), linear(i, 420, 300, -3, 0)]
            out = tr.update(i, dets)
            seq.append(tuple((t.track_id, round(t.box.x1)) for t in out))
        return seq

    def test_deterministic(self) -> None:
        assert self._run() == self._run()

    def test_reset_restarts_ids(self, tracker: FaceTracker) -> None:
        for i in range(5):
            tracker.update(i, [face(150, 240)])
        tracker.update(5, [face(150, 240)])
        tracker.reset()
        assert len(tracker) == 0
        out = tracker.update(0, [face(150, 240)])
        assert {t.track_id for t in out} == {1}

    def test_len_and_active_ids_track_state(self, tracker: FaceTracker) -> None:
        for i in range(4):
            tracker.update(i, [face(100, 100), face(500, 400)])
        assert len(tracker) == 2
        assert tracker.active_track_ids == sorted(tracker.active_track_ids)


# --------------------------------------------------------------------------- #
# Landmarks passthrough                                                        #
# --------------------------------------------------------------------------- #
class TestLandmarks:
    def test_landmarks_survive_tracking(self, tracker: FaceTracker) -> None:
        lm = np.arange(10, dtype=np.float32).reshape(5, 2)
        tracker.update(0, [Detection(*face(200, 200).xyxy(), 0.9, landmarks=lm)])
        out = tracker.update(1, [Detection(*face(202, 200).xyxy(), 0.9, landmarks=lm)])
        assert len(out) == 1
        assert out[0].landmarks is not None
        np.testing.assert_array_equal(out[0].landmarks, lm)

    def test_coasted_face_has_no_landmarks(self, tracker: FaceTracker) -> None:
        lm = np.zeros((5, 2), dtype=np.float32)
        tracker.update(0, [Detection(*face(200, 200).xyxy(), 0.9, landmarks=lm)])
        tracker.update(1, [Detection(*face(206, 200).xyxy(), 0.9, landmarks=lm)])
        out = tracker.update(2, None)
        assert out and out[0].coasting
        assert out[0].landmarks is None
