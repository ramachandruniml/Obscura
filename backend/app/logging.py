"""Structured logging setup (structlog).

Every pipeline stage emits one event with a stable ``event`` name and a
``job_id`` binding so a single job can be grepped end to end:

    upload.received  -> detection.started -> detection.completed (n_faces=...)
    -> tracking.completed (n_tracks=...) -> redaction.applied -> job.completed

Call ``configure_logging()`` once at process start (API and worker both do).
Use ``get_logger(__name__)`` everywhere else.
"""

from __future__ import annotations

import logging
import sys

import structlog

from app.config import settings

_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)


def bind_job(job_id: str) -> None:
    """Bind job_id to the context so every subsequent log line carries it."""
    structlog.contextvars.bind_contextvars(job_id=job_id)


def clear_job() -> None:
    structlog.contextvars.clear_contextvars()
