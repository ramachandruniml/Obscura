"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes_jobs import router as jobs_router
from app.api.routes_redact import router as redact_router
from app.config import settings
from app.errors import ObscuraError
from app.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "api.startup",
        env=settings.obscura_env,
        storage_dir=str(settings.storage_dir),
        detector_backend=settings.detector_backend,
        eager=settings.celery_task_always_eager,
    )
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
    allow_origins=settings.cors_origins,  # explicit list, never "*"
    allow_credentials=False,  # no cookies / auth headers are used
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=600,
)


@app.exception_handler(ObscuraError)
async def _obscura_error_handler(request: Request, exc: ObscuraError) -> JSONResponse:
    log.info("api.error", path=request.url.path, status=exc.http_status, detail=str(exc))
    return JSONResponse(status_code=exc.http_status, content={"detail": str(exc)})


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


app.include_router(redact_router)
app.include_router(jobs_router)
