"""Job runner: manifest -> detector + redactor -> the right pipeline -> manifest.

Called by the Celery task (and directly by tests in eager mode). Owns the
job status transitions and the structured ``job_id``-bound logging context.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime

from app import storage
from app.config import settings
from app.detectors import get_cached_detector
from app.errors import JobNotFoundError, ObscuraError
from app.logging import bind_job, clear_job, get_logger
from app.pipeline.image_pipeline import redact_image
from app.pipeline.video_pipeline import redact_video
from app.redaction.redactors import build_redactor

log = get_logger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def process_job(job_id: str) -> dict:
    """Run the redaction job identified by ``job_id``. Returns the final manifest."""
    manifest = storage.read_manifest(job_id)
    if manifest is None:
        raise JobNotFoundError(job_id)

    bind_job(job_id)
    try:
        storage.update_manifest(job_id, status="processing", started_at=_now())
        src = storage.input_path(job_id)
        if src is None or not src.exists():
            raise JobNotFoundError(f"input file missing for job {job_id}")

        detector = get_cached_detector()
        redactor = build_redactor(manifest["method"])
        confidence = float(manifest["confidence"])
        job_path = storage.job_dir(job_id)

        if manifest["kind"] == "image":
            dst = job_path / f"output{src.suffix}"
            result = redact_image(src, dst, detector, redactor, confidence)
        else:
            dst = job_path / "output.mp4"
            result = redact_video(src, dst, detector, redactor, confidence)

        final = storage.update_manifest(
            job_id,
            status="complete",
            finished_at=_now(),
            output_name=dst.name,
            stats=asdict(result),
        )
        if settings.delete_input_on_success:
            storage.delete_input(job_id)
        log.info("job.done", status="complete", **asdict(result))
        return final

    except ObscuraError as exc:
        log.warning("job.failed", error=str(exc), kind="expected")
        return storage.update_manifest(job_id, status="failed", error=str(exc), finished_at=_now())
    except Exception as exc:
        log.exception("job.failed", kind="unexpected")
        storage.update_manifest(
            job_id, status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=_now()
        )
        raise
    finally:
        clear_job()
