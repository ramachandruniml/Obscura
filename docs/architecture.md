# Architecture

> Skeleton — expanded per deliverable. The prose diagram description for the
> README is finalized in Deliverable 10.

## Services (Docker Compose)

| Service | Image / build | Role |
|---|---|---|
| `redis` | `redis:7-alpine` | Celery broker (db 0) + result backend (db 1) |
| `api` | `backend/` | FastAPI. Validates uploads, writes input to the shared volume, enqueues a job, serves status + results. Never runs the model in-process. |
| `worker` | `backend/` (same image) | Celery worker. Runs the detect → track → redact pipeline. `--concurrency=1` because inference is CPU-bound. |
| `beat` | `backend/` (same image) | Celery beat. Fires `sweep_expired_artifacts` every `CLEANUP_INTERVAL_SECONDS`. |
| `frontend` | `frontend/` | React build served by nginx; proxies `/api/` → `api:8000`. |

Shared state: one named volume `obscura-data` mounted at `/data` in api, worker, beat.

## Request flow

```
browser ──POST /api/redact (multipart)──▶ nginx ──▶ api
   api: validate type/size/duration ─▶ write /data/<job_id>/input.<ext>
        ─▶ create job record (Redis) ─▶ enqueue Celery task ─▶ 202 {job_id}
   worker: load detector (cached) ─▶ decode ─▶ detect (+ByteTrack for video)
        ─▶ redact each tracked box (+padding) ─▶ encode ─▶ /data/<job_id>/output.<ext>
        ─▶ mark job complete (result URL)
   browser: poll GET /api/jobs/<job_id> until complete ─▶ GET result ─▶ download
   beat: delete /data/<job_id>/ once older than RESULT_TTL_SECONDS
```

## Detector interface

`app/detectors/base.py` defines `Detector.detect(frame: np.ndarray, conf: float) -> list[Detection]`.
Implementations: `RetinaFaceDetector` (biubug6 Pytorch_Retinaface), `YOLOv8FaceDetector`
(Ultralytics), `OnnxFaceDetector` (ONNX Runtime, backend-agnostic). Selection via
`app/detectors/registry.py` keyed on `DETECTOR_BACKEND` + `DETECTOR_RUNTIME`.

## Privacy properties

- No recognition / embeddings / identity matching anywhere in the codebase.
- Frames exist only in worker memory during a job.
- Inputs and outputs auto-deleted after `RESULT_TTL_SECONDS` (default 1h).
- No analytics, no third-party calls from the worker.
