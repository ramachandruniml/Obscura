"""FastAPI application entrypoint.

Skeleton for now: health + readiness only. The /redact and /jobs/{id} routers
are wired in Deliverable 5.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    log.info("api.startup", env=settings.obscura_env, storage_dir=str(settings.storage_dir))
    yield
    log.info("api.shutdown")


app = FastAPI(
    title="Obscura",
    version="0.1.0",
    summary="Automatic face detection + redaction. No recognition, no retention.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz", tags=["meta"])
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz", tags=["meta"])
async def readyz() -> dict[str, object]:
    return {
        "status": "ok",
        "detector_backend": settings.detector_backend,
        "detector_runtime": settings.detector_runtime,
        "device": settings.device,
    }


# Deliverable 5:
# from app.api.routes_redact import router as redact_router
# from app.api.routes_jobs import router as jobs_router
# app.include_router(redact_router)
# app.include_router(jobs_router)
