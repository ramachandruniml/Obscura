"""Celery application: broker/backend wiring, task defaults, and the TTL beat job.

Set ``CELERY_TASK_ALWAYS_EAGER=true`` to run jobs inline in the calling process
(used by the API tests, and usable for a broker-less local run).
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
    task_always_eager=settings.celery_task_always_eager,
    task_eager_propagates=True,
    beat_schedule={
        "sweep-expired-artifacts": {
            "task": "app.worker.tasks.sweep_expired_artifacts",
            "schedule": float(settings.cleanup_interval_seconds),
        },
    },
)
