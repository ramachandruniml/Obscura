"""Local-filesystem job storage + retention sweep.

Layout under ``settings.storage_dir``::

    <job_id>/
        input.<ext>       (deleted on successful completion)
        output.<ext>
        manifest.json     (single source of truth for job status)

``manifest.json`` is written atomically (temp file + os.replace) so the API and
worker containers can read/write it over the shared volume without tearing.
The sweep deletes any job directory whose manifest ``expires_at`` has passed
(or that has no readable manifest and is older than the TTL).
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.config import settings
from app.errors import JobNotFoundError
from app.logging import get_logger

log = get_logger(__name__)

MANIFEST_NAME = "manifest.json"


def new_job_id() -> str:
    return uuid.uuid4().hex


def _root() -> Path:
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings.storage_dir


def job_dir(job_id: str) -> Path:
    return _root() / job_id


def create_job_dir(job_id: str) -> Path:
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=False)
    return d


def _manifest_path(job_id: str) -> Path:
    return job_dir(job_id) / MANIFEST_NAME


def write_manifest(job_id: str, data: dict[str, Any]) -> None:
    path = _manifest_path(job_id)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_manifest(job_id: str) -> dict[str, Any] | None:
    path = _manifest_path(job_id)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def update_manifest(job_id: str, **changes: Any) -> dict[str, Any]:
    data = read_manifest(job_id)
    if data is None:
        raise JobNotFoundError(job_id)
    data.update(changes)
    write_manifest(job_id, data)
    return data


def init_manifest(
    job_id: str, *, kind: str, method: str, confidence: float, input_name: str
) -> dict[str, Any]:
    now = datetime.now(UTC)
    data = {
        "job_id": job_id,
        "status": "pending",
        "kind": kind,
        "method": method,
        "confidence": confidence,
        "input_name": input_name,
        "output_name": None,
        "error": None,
        "stats": {},
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=settings.result_ttl_seconds)).isoformat(),
    }
    write_manifest(job_id, data)
    return data


def input_path(job_id: str) -> Path | None:
    for p in sorted(job_dir(job_id).glob("input.*")):
        return p
    return None


def result_path(job_id: str) -> Path | None:
    data = read_manifest(job_id)
    if not data or not data.get("output_name"):
        return None
    p = job_dir(job_id) / data["output_name"]
    return p if p.exists() else None


def delete_input(job_id: str) -> None:
    p = input_path(job_id)
    if p and p.exists():
        p.unlink()
        log.info("storage.input_deleted", job_id=job_id)


def purge_job(job_id: str) -> None:
    shutil.rmtree(job_dir(job_id), ignore_errors=True)


def sweep(now: float | None = None) -> int:
    """Delete expired job directories. Returns the number removed."""
    now = time.time() if now is None else now
    root = _root()
    removed = 0
    for d in root.iterdir():
        if not d.is_dir():
            continue
        data = read_manifest(d.name)
        expired: bool
        if data and data.get("expires_at"):
            try:
                expired = datetime.fromisoformat(data["expires_at"]).timestamp() < now
            except ValueError:
                expired = d.stat().st_mtime < now - settings.result_ttl_seconds
        else:
            # no usable manifest: fall back to directory age
            expired = d.stat().st_mtime < now - settings.result_ttl_seconds
        if expired:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    if removed:
        log.info("storage.sweep", removed=removed)
    return removed
