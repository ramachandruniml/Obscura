# Obscura

[![ci](https://github.com/ramachandruniml/Obscura/actions/workflows/ci.yml/badge.svg)](https://github.com/ramachandruniml/Obscura/actions/workflows/ci.yml)

Automatic face **detection + redaction** for images and video, so footage can be
shared publicly without exposing bystanders. Blur, pixelate, or black-box every
face; keep IDs stable across video frames so redaction doesn't flicker.

**Not** face recognition. No identity matching, no embeddings, no stored frames.
Uploads and results auto-delete after a configurable TTL (default 1 hour).

> Status: **Deliverable 9/10 complete** — GitHub Actions CI.
> See the deliverable checklist below.

---

## Deliverable checklist

- [x] 1. Project structure + Docker Compose skeleton
- [x] 2. Detector interface + RetinaFace + YOLOv8-face
- [x] 3. Tracking module (ByteTrack)
- [x] 4. Redaction module (blur / pixelate / box)
- [x] 5. FastAPI endpoints + Celery job queue
- [x] 6. Benchmark script + WIDER FACE report
- [x] 7. ONNX export + latency comparison report
- [x] 8. React frontend
- [x] 9. GitHub Actions CI
- [ ] 10. README finalization + architecture diagram + benchmark results

## CI

`.github/workflows/ci.yml` runs on every push and PR:

| Job | Steps |
|---|---|
| `backend` | `uv` install (CPU torch) → `ruff check` + `ruff format --check` → `pytest` (with `--cov=app`) |
| `frontend` | `npm ci` → `eslint` → `tsc --noEmit` → `vitest run` → `vite build` |
| `docker` | builds the `backend/` and `frontend/` images (gated on the two jobs above; GHA layer cache) |

---

## Quick start (Docker Compose)

```bash
cp .env.example .env
# Populate ./backend/weights/ (YOLOv8-face auto-downloads; RetinaFace needs a URL/gdrive-id):
#   cd backend && uv run python scripts/fetch_weights.py
docker compose up --build
```

| URL | What |
|---|---|
| http://localhost:3000 | Frontend |
| http://localhost:8000/docs | API (OpenAPI / Swagger) |
| http://localhost:8000/healthz | Health check |

### API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/redact` | multipart: `file` (image/video) + `method` (`blur`\|`pixelate`\|`box`) + `confidence` (0–1). Returns `202 {job_id, status}`. |
| `GET` | `/jobs/{id}` | Job status (`pending`\|`processing`\|`complete`\|`failed`), `stats`, and `result_url` when done. |
| `GET` | `/jobs/{id}/result` | The redacted file (image keeps its format; video is always `.mp4`). `404` until complete or after TTL. |

Errors: `415` unsupported type, `413` too large, `422` corrupt / video too long, `400` bad method/confidence, `404` unknown or expired job.

Without Docker you can run the whole thing in one process — set `CELERY_TASK_ALWAYS_EAGER=true` and `pip install -e ".[dev]"`, then `uvicorn app.main:app`.

GPU (needs NVIDIA Container Toolkit + a CUDA torch build — see `backend/Dockerfile`
`BUILD_TORCH_VARIANT`):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Local dev (without Docker)

```bash
# backend
cd backend
uv venv && uv pip install ".[dev]"
uv run uvicorn app.main:app --reload
uv run celery -A app.worker.celery_app.celery worker -B --loglevel=INFO   # worker + beat

# frontend
cd frontend
npm install
npm run dev        # proxies /api -> http://localhost:8000
```

## Configuration

Every adjustable value lives in `.env` (read by `backend/app/config.py`). Notable knobs:

| Var | Default | Meaning |
|---|---|---|
| `MAX_UPLOAD_BYTES` | 104857600 | Reject uploads larger than this |
| `ALLOWED_IMAGE_TYPES` / `ALLOWED_VIDEO_TYPES` | jpeg/png/webp, mp4/mov/webm | Accepted MIME types |
| `RESULT_TTL_SECONDS` | 3600 | Auto-delete inputs + outputs after this |
| `DETECTOR_BACKEND` | `retinaface` | `retinaface` \| `yolov8face` |
| `DETECTOR_RUNTIME` | `pytorch` | `pytorch` \| `onnx` |
| `DEVICE` | `cpu` | `cpu` \| `cuda` |
| `DEFAULT_CONFIDENCE` | 0.5 | Detection score threshold |
| `DEFAULT_REDACTION` | `blur` | `blur` \| `pixelate` \| `box` |
| `BOX_PADDING_RATIO` | 0.15 | Extra padding around every redacted box |
| `DETECT_EVERY_N_FRAMES` | 5 | Detector cadence for video; tracker fills the gaps |

## Project layout

```
obscura/
├── docker-compose.yml            # api + worker + beat + redis + frontend
├── docker-compose.gpu.yml        # NVIDIA override
├── .env.example                  # every adjustable knob
├── backend/
│   ├── app/
│   │   ├── config.py             # pydantic-settings — single source of knobs
│   │   ├── logging.py            # structlog — per-stage structured logs
│   │   ├── main.py               # FastAPI app
│   │   ├── api/                  # routes: /redact, /jobs/{id}      (D5)
│   │   ├── detectors/            # Detector ABC + RetinaFace + YOLOv8-face  (D2)
│   │   ├── tracking/             # ByteTrack wrapper                 (D3)
│   │   ├── redaction/            # blur / pixelate / box            (D4)
│   │   ├── pipeline/             # image + video pipelines          (D5)
│   │   └── worker/               # celery app, tasks, TTL sweep     (D5)
│   ├── scripts/                  # benchmark_widerface, export_onnx (D6, D7)
│   ├── weights/                  # downloaded / exported models (git-ignored)
│   └── tests/                    # detection / tracking / redaction / api
├── frontend/                     # React + TS + Tailwind + Vite     (D8)
├── docs/
│   ├── architecture.md
│   └── benchmarks/               # generated reports (git-ignored)
└── .github/workflows/ci.yml      # lint + test both sides           (D9)
```

## Architecture

See [docs/architecture.md](docs/architecture.md). Prose diagram description
finalized in Deliverable 10.

## Privacy / data retention

- No face recognition, identity matching, or embedding storage anywhere.
- Frames are held only in worker memory for the duration of a job.
- Inputs and outputs are deleted after `RESULT_TTL_SECONDS` by the `beat` sweep.
- The worker makes no outbound network calls.

## Benchmark results

<!-- Fill in after running `scripts/benchmark_widerface.py` and `scripts/export_onnx.py`. -->

### WIDER FACE (validation) — detector comparison

| Detector | Runtime | Precision | Recall | Inference FPS | Notes |
|---|---|---|---|---|---|
| RetinaFace (MobileNet0.25) | PyTorch | _TBD_ | _TBD_ | _TBD_ | |
| RetinaFace (ResNet50) | PyTorch | _TBD_ | _TBD_ | _TBD_ | |
| YOLOv8-face (n) | PyTorch | _TBD_ | _TBD_ | _TBD_ | |

_Hardware: TBD. Command: `uv run python scripts/benchmark_widerface.py --data-root <path>`._

### PyTorch vs ONNX Runtime — latency

| Model | Runtime | Mean latency (ms) | p95 (ms) | FPS | Speedup |
|---|---|---|---|---|---|
| _selected detector_ | PyTorch (CPU) | _TBD_ | _TBD_ | _TBD_ | 1.00× |
| _selected detector_ | ONNX Runtime (CPU) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

_Hardware: TBD. Command: `uv run python scripts/export_onnx.py --backend <name>`._

## License / intended use

Built for redacting bystanders in footage you have the right to share. Not for
surveillance, identification, or tracking individuals.
