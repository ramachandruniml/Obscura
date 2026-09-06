"""Pydantic request/response models for the API."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    pending = "pending"
    processing = "processing"
    complete = "complete"
    failed = "failed"


class RedactAccepted(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.pending


class JobResource(BaseModel):
    job_id: str
    status: JobStatus
    kind: str | None = None
    method: str
    confidence: float
    input_name: str | None = None
    created_at: str
    expires_at: str
    result_url: str | None = None
    error: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_manifest(
        cls, manifest: dict[str, Any], *, result_url: str | None = None
    ) -> JobResource:
        return cls(
            job_id=manifest["job_id"],
            status=JobStatus(manifest["status"]),
            kind=manifest.get("kind"),
            method=manifest["method"],
            confidence=manifest["confidence"],
            input_name=manifest.get("input_name"),
            created_at=manifest["created_at"],
            expires_at=manifest["expires_at"],
            result_url=result_url,
            error=manifest.get("error"),
            stats=manifest.get("stats") or {},
        )


class ErrorResponse(BaseModel):
    detail: str
