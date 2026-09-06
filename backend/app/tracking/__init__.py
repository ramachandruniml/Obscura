"""Multi-face tracking across video frames.

Deliverable 3:
    tracker.py - ByteTrack wrapper (via `supervision`). Assigns persistent
    integer IDs to faces so redaction is consistent frame to frame, and lets
    the pipeline detect only every Nth frame (settings.detect_every_n_frames)
    while tracking coasts the boxes in between.
"""
