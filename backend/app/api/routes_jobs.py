"""GET /jobs/{id} — status + result URL; GET /jobs/{id}/result — the file."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app import storage
from app.errors import JobNotFoundError
from app.schemas import JobResource

router = APIRouter(tags=["jobs"])

_RESULT_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
}


@router.get("/jobs/{job_id}", response_model=JobResource)
async def get_job(job_id: str) -> JobResource:
    manifest = storage.read_manifest(job_id)
    if manifest is None:
        raise JobNotFoundError(job_id)
    result_url = f"/jobs/{job_id}/result" if manifest["status"] == "complete" else None
    return JobResource.from_manifest(manifest, result_url=result_url)


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> FileResponse:
    manifest = storage.read_manifest(job_id)
    if manifest is None:
        raise JobNotFoundError(job_id)
    if manifest["status"] != "complete":
        raise JobNotFoundError(f"job {job_id} is {manifest['status']}, no result available")

    path = storage.result_path(job_id)
    if path is None:
        raise JobNotFoundError(f"result for job {job_id} is gone (expired)")

    media_type = _RESULT_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    stem = (manifest.get("input_name") or f"redacted{path.suffix}").rsplit(".", 1)[0]
    return FileResponse(path, media_type=media_type, filename=f"{stem}_redacted{path.suffix}")
