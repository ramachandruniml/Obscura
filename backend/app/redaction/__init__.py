"""Redaction methods.

Public surface:
    Redactor                 - ABC; .apply(frame, regions, pad_ratio=, copy=)
    GaussianBlurRedactor, PixelateRedactor, SolidBoxRedactor
    build_redactor(method)   - method name -> Redactor (defaults to settings)
    get_redactor_class(method), REDACTION_METHODS
"""

from app.redaction.redactors import (
    REDACTION_METHODS,
    GaussianBlurRedactor,
    PixelateRedactor,
    Redactor,
    SolidBoxRedactor,
    build_redactor,
    get_redactor_class,
)

__all__ = [
    "REDACTION_METHODS",
    "GaussianBlurRedactor",
    "PixelateRedactor",
    "Redactor",
    "SolidBoxRedactor",
    "build_redactor",
    "get_redactor_class",
]
