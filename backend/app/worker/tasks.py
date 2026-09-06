"""Celery tasks.

Skeleton: only the TTL sweep is wired so `worker` and `beat` boot cleanly.
`run_image_job` / `run_video_job` are implemented in Deliverable 5 once the
pipeline modules exist.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.config import settings
from app.logging import get_logger
from app.worker.celery_app import celery

log = get_logger(__name__)


@celery.task(name="app.worker.tasks.sweep_expired_artifacts")
def sweep_expired_artifacts() -> dict[str, int]:
    """Delete files/dirs under STORAGE_DIR older than result_ttl_seconds.

    Full implementation (job-manifest aware) lands with app/storage.py in
    Deliverable 5. This version is a plain mtime sweep so retention works today.
    """
    root: Path = settings.storage_dir
    if not root.exists():
        return {"removed": 0}

    cutoff = time.time() - settings.result_ttl_seconds
    removed = 0
    for entry in root.iterdir():
        try:
            if entry.stat().st_mtime < cutoff:
                if entry.is_dir():
                    for sub in sorted(entry.rglob("*"), reverse=True):
                        sub.unlink() if sub.is_file() else sub.rmdir()
                    entry.rmdir()
                else:
                    entry.unlink()
                removed += 1
        except OSError as exc:  # noqa: PERF203 - best-effort sweep
            log.warning("cleanup.skip", path=str(entry), error=str(exc))

    log.info("cleanup.sweep", removed=removed, ttl_seconds=settings.result_ttl_seconds)
    return {"removed": removed}
