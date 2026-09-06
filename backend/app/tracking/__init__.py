"""Multi-face tracking across video frames.

Public surface:
    FaceTracker  - stateful per-video tracker (ByteTrack + coasting)
    TrackedFace  - one face with a persistent track_id for a frame
"""

from app.tracking.tracker import FaceTracker, TrackedFace

__all__ = ["FaceTracker", "TrackedFace"]
