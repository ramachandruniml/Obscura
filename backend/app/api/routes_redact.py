"""POST /redact — accept an image or video, enqueue a redaction job."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile

from app import media, storage
from app.config import settings
from app.errors import ObscuraError
from app.logging import get_logger
from app.redaction.redactors import REDACTION_METHODS
from app.schemas import RedactAccepted
from app.worker.tasks import run_redaction_job

log = get_logger(__name__)
router = APIRouter(tags=["redaction"])


@router.post("/redact", response_model=RedactAccepted, status_code=202)
async def redact(
    file: UploadFile = File(..., description="image (jpeg/png/webp) or video (mp4/mov/webm)"),
    method: str = Form(settings.default_redaction),
    confidence: float = Form(settings.default_confidence),
) -> RedactAccepted:
    if method not in REDACTION_METHODS:
        raise ObscuraError(f"unknown method {method!r}; valid: {', '.join(REDACTION_METHODS)}")
    if not 0.0 <= confidence <= 1.0:
        raise ObscuraError(f"confidence must be in [0, 1], got {confidence}")

    kind, ext = media.classify(file.content_type, file.filename)

    job_id = storage.new_job_id()
    storage.create_job_dir(job_id)
    try:
        dst = storage.job_dir(job_id) / f"input{ext}"
        size = media.stream_to_file(file.file, dst)
        probe = media.probe(dst, kind)
    except ObscuraError:
        storage.purge_job(job_id)
        raise

    manifest = storage.init_manifest(
        job_id,
        kind=kind,
        method=method,
        confidence=confidence,
        input_name=file.filename or f"input{ext}",
    )
    manifest["stats"] = {"upload_bytes": size, **probe}
    storage.write_manifest(job_id, manifest)

    log.info(
        "upload.received",
        job_id=job_id,
        kind=kind,
        method=method,
        confidence=confidence,
        bytes=size,
        **probe,
    )
    run_redaction_job.delay(job_id)
    return RedactAccepted(job_id=job_id)
