"""Redaction methods.

Deliverable 4:
    redactors.py - Redactor ABC + GaussianBlur / Pixelate / SolidBox.
    Each takes a frame + list of boxes, expands every box by
    settings.box_padding_ratio (more for coasted tracks), clamps to frame
    bounds, and applies the effect. Pure functions of (frame, boxes, params).
"""
