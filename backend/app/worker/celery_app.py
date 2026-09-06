"""Celery application.

Skeleton for now: app object + beat schedule for TTL cleanup. The redaction
tasks (``run_image_job`` / ``run_video_job``) are added in Deliverable 5; the
cleanup task body lands with the storage module (Deliverable 5) too.
"""

from __future__ import annotations

from celery import Celery

from app.config import settings
from app.logging import configure_logging

configure_logging()

celery = Celery(
    "obscura",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.worker.tasks"],
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_time_limit=60 * 30,
    task_soft_time_limit=60 * 25,
    worker_max_tasks_per_child=20,
    result_expires=settings.result_ttl_seconds,
    beat_schedule={
        "sweep-expired-artifacts": {
            "task": "app.worker.tasks.sweep_expired_artifacts",
            "schedule": float(settings.cleanup_interval_seconds),
        },
    },
)
