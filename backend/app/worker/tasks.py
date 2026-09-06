"""Celery tasks: run a redaction job, and sweep expired artifacts."""

from __future__ import annotations

from app import storage
from app.logging import get_logger
from app.worker.celery_app import celery

log = get_logger(__name__)


@celery.task(name="app.worker.tasks.run_redaction_job", max_retries=0)
def run_redaction_job(job_id: str) -> dict:
    from app.pipeline.runner import process_job

    return process_job(job_id)


@celery.task(name="app.worker.tasks.sweep_expired_artifacts")
def sweep_expired_artifacts() -> dict[str, int]:
    removed = storage.sweep()
    return {"removed": removed}
