# Obscura

[![ci](https://github.com/ramachandruniml/Obscura/actions/workflows/ci.yml/badge.svg)](https://github.com/ramachandruniml/Obscura/actions/workflows/ci.yml)

Automatic face **detection + redaction** for images and video, so footage can be
shared publicly without exposing bystanders. Blur, pixelate, or black-box every
face; ByteTrack keeps face IDs stable across video frames so the redaction
doesn't flicker and the detector only runs every Nth frame.

**Not** face recognition. No identity matching, no embeddings, no stored frames.
Uploaded files and results auto-delete after a configurable TTL (default 1 hour),
and the worker makes no outbound network calls.

Built in 10 stages — see [Project layout](#project-layout) for the map. All
backend + frontend tests and both Docker image builds pass in
[CI](.github/workflows/ci.yml).

---

## What you need to do to run it

**TL;DR:** install Docker, copy the env file, get **one** set of detector
weights, `docker compose up`.

### 1. Prerequisites

| To run with… | You need |
|---|---|
| **Docker** (recommended) | Docker Desktop / Engine with Compose v2. Ports **3000**, **8000**, **6379** free. |
| **No Docker** | Python **3.11** (not 3.12+), Node **20**, `ffmpeg` on PATH (or rely on the bundled `imageio-ffmpeg`), a local Redis, and [`uv`](https://docs.astral.sh/uv/). |
| **GPU** | NVIDIA Container Toolkit **and** a CUDA torch build in the image (`--build-arg BUILD_TORCH_VARIANT=cu121`). CPU is the default everywhere. |

### 2. Configuration

```bash
cp .env.example .env
```

The defaults work as-is. Every knob is documented in `.env.example` and read by
`backend/app/config.py`.

### 3. Model weights  ⚠️ **the one manual step**

Weights are **not** committed and **not** fully auto-downloaded. Put them in
`backend/weights/` (mounted into the `api` and `worker` containers).

```bash
cd backend
uv run python scripts/fetch_weights.py            # or: python scripts/fetch_weights.py
```

- **YOLOv8-face** — the script tries to download `yolov8n-face.pt` from the
  `akanametov/yolo-face` releases. If that URL 404s, grab any YOLOv8-face `.pt`
  (e.g. from a HuggingFace mirror) and save it as `backend/weights/yolov8n-face.pt`,
  or pass `--yolov8-url <url>`.
- **RetinaFace** — Google-Drive hosted by the upstream repo
  ([biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)),
  so the script won't fetch it blind. Either:
  - `python scripts/fetch_weights.py --only retinaface --retinaface-url <direct url>`, or
  - `--retinaface-gdrive-id <id>` (needs `pip install gdown`), or
  - download `mobilenet0.25_Final.pth` manually and save it as
    `backend/weights/retinaface_mobilenet0.25.pth`.

**`DETECTOR_BACKEND` defaults to `retinaface`.** If you only have YOLOv8-face
weights, set `DETECTOR_BACKEND=yolov8face` in `.env` — otherwise every job fails
with "RetinaFace weights not found".

> The vendored RetinaFace decode path (`backend/app/detectors/_retinaface/`) is a
> faithful port of the upstream model but has not yet been run against the real
> checkpoint in this repo. If RetinaFace results look wrong on your first job,
> switch to `DETECTOR_BACKEND=yolov8face` (routed through Ultralytics, lower risk)
> while it's verified.

### 4. Run

```bash
docker compose up --build
```

| URL | What |
|---|---|
| <http://localhost:3000> | Frontend (upload → redact → before/after → download) |
| <http://localhost:8000/docs> | API — OpenAPI / Swagger UI |
| <http://localhost:8000/healthz> | Liveness |
| <http://localhost:8000/readyz> | Shows the active detector backend / runtime / device |

GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

### 5. (Optional) benchmarks + ONNX

Only needed to fill in the [Benchmark results](#benchmark-results) tables.

- **WIDER FACE** — download `WIDER_val.zip` + `wider_face_split.zip` (~1.8 GB)
  from <http://shuoyang1213.me/WIDERFACE/> into `datasets/widerface/`, then
  `cd backend && uv run python scripts/benchmark_widerface.py --data-root ../datasets/widerface`.
- **ONNX latency** — `cd backend && uv run python scripts/export_onnx.py`
  (exports YOLOv8-face, times PyTorch vs ONNX Runtime, writes
  `docs/benchmarks/onnx-latency.md`). To *serve* inference from ONNX, set
  `DETECTOR_RUNTIME=onnx` once `weights/detector.onnx` exists.

---

## Architecture

Five containers, one shared volume. Full write-up in
[docs/architecture.md](docs/architecture.md).

```
                 ┌─────────────┐        ┌──────────────────────────────┐
  browser ─POST /api/redact──▶ │  nginx  │ ──/api──▶ │            api (FastAPI)              │
   (React SPA)  ◀──202 job_id──│ frontend│ ◀───────  │  validate type/size/duration/pixels  │
        │                      └─────────┘           │  write /data/<job>/input.<ext>        │
        │  poll GET /api/jobs/<id>                    │  init manifest.json · enqueue task    │
        │                                             └───────┬──────────────┬───────────────┘
        │                                                     │ Celery       │ reads
        │                                             ┌───────▼──────┐   ┌───▼───────────────┐
        ▼                                             │    redis     │   │  obscura-data vol │
  GET /api/jobs/<id>/result ──▶ (redacted file)       │ broker+result│   │  /data/<job_id>/  │
                                                      └───────┬──────┘   │   input.<ext>     │
                                                              │ dequeue  │   output.<ext>    │
                                              ┌───────────────▼───────┐  │   manifest.json   │
                                              │       worker          │  └───────────────────┘
                                              │  image:  decode → detect → redact → encode
                                              │  video:  demux → (detect every Nth frame +
                                              │          ByteTrack coast) → redact → x264 →
                                              │          ffmpeg audio remux
                                              │  manifest: processing → complete/failed
                                              │  delete input on success
                                              └───────────────────────┘
                                              ┌───────────────────────┐
                                              │   beat  (Celery beat) │  every CLEANUP_INTERVAL_SECONDS:
                                              │                       │  delete /data/<job>/ past manifest.expires_at
                                              └───────────────────────┘
```

**Detector interface** — `app/detectors/base.py` defines `Detection` (xyxy pixel
box + score + optional 5-pt landmarks) and the `Detector` ABC. Implementations —
`RetinaFaceDetector` (vendored biubug6 Pytorch_Retinaface), `YOLOv8FaceDetector`
(Ultralytics), `OnnxFaceDetector` (ONNX Runtime) — are interchangeable; the
registry picks one from `DETECTOR_BACKEND` + `DETECTOR_RUNTIME`.

**Redaction** — `Redactor` ABC + `GaussianBlur` / `Pixelate` / `SolidBox`. Each
expands the box by `BOX_PADDING_RATIO` (more for coasted tracks) before applying
the effect, so edges aren't left exposed.

**Structured logs** — every job binds `job_id` and emits one event per stage:
`upload.received → detection.started → detection.completed (n_faces) →
tracking.completed (n_tracks) → redaction.applied → job.completed`.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/redact` | multipart form: `file` (image or video) + `method` (`blur`\|`pixelate`\|`box`) + `confidence` (0–1). Returns `202 {job_id, status}` and processes in the background. |
| `GET` | `/jobs/{id}` | Job status (`pending`\|`processing`\|`complete`\|`failed`), `stats` (`n_faces`, dimensions, frame counts…), `error`, and `result_url` once complete. |
| `GET` | `/jobs/{id}/result` | The redacted file — images keep their format, video is always `.mp4`. `404` before completion or after the TTL. |

Error codes: `415` unsupported type · `413` too large · `422` corrupt file or
video too long · `400` bad method/confidence · `404` unknown or expired job.

Broker-less single-process mode (handy for a quick local try): set
`CELERY_TASK_ALWAYS_EAGER=true`, `pip install -e "backend[dev]"`, then
`uvicorn app.main:app` — jobs run inline in the request.

---

## Configuration

Full list in `.env.example`. The ones you're most likely to touch:

| Var | Default | Meaning |
|---|---|---|
| `DETECTOR_BACKEND` | `retinaface` | `retinaface` \| `yolov8face` |
| `DETECTOR_RUNTIME` | `pytorch` | `pytorch` \| `onnx` (needs `weights/detector.onnx`) |
| `DEVICE` | `cpu` | `cpu` \| `cuda` |
| `DEFAULT_REDACTION` | `blur` | `blur` \| `pixelate` \| `box` |
| `DEFAULT_CONFIDENCE` | `0.5` | detection score threshold |
| `BOX_PADDING_RATIO` | `0.15` | padding added around every redacted box |
| `DETECT_EVERY_N_FRAMES` | `5` | detector cadence for video; ByteTrack fills the gaps |
| `MAX_UPLOAD_BYTES` | `104857600` | reject uploads larger than this |
| `ALLOWED_IMAGE_TYPES` / `ALLOWED_VIDEO_TYPES` | jpeg/png/webp · mp4/mov/webm | accepted MIME types |
| `MAX_VIDEO_DURATION_SECONDS` | `300` | reject longer videos |
| `RESULT_TTL_SECONDS` | `3600` | auto-delete inputs + outputs after this |

---

## Local dev (without Docker)

```bash
# --- backend ---
cd backend
uv venv && uv pip install ".[dev]"
uv run python scripts/fetch_weights.py                 # weights (see step 3 above)
# start a Redis somewhere, then:
uv run uvicorn app.main:app --reload
uv run celery -A app.worker.celery_app.celery worker -B --loglevel=INFO   # worker + beat

# --- frontend ---
cd frontend
npm ci
npm run dev        # http://localhost:5173, proxies /api -> http://localhost:8000
```

Checks (what CI runs):

```bash
cd backend  && ruff check . && ruff format --check . && pytest -q
cd frontend && npm run lint && npm run typecheck && npm run test && npm run build
```

Tests need **no** model weights or dataset — detection/tracking/redaction/API are
exercised with a `FakeDetector`; weight-gated integration tests skip themselves
when the files are absent.

---

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
│   │   ├── errors.py             # ObscuraError hierarchy -> HTTP status
│   │   ├── media.py              # upload classify / size cap / probe
│   │   ├── storage.py            # per-job dir + manifest.json + TTL sweep
│   │   ├── schemas.py            # request/response models
│   │   ├── main.py               # FastAPI app + exception handler
│   │   ├── api/                  # POST /redact, GET /jobs/{id}[/result]
│   │   ├── detectors/            # Detector ABC + RetinaFace + YOLOv8-face + ONNX
│   │   │   └── _retinaface/      # vendored biubug6 model (NOTICE + LICENSE)
│   │   ├── tracking/             # FaceTracker: ByteTrack + coasting
│   │   ├── redaction/            # Redactor ABC + blur / pixelate / box
│   │   ├── pipeline/             # image_pipeline, video_pipeline, runner
│   │   ├── worker/               # celery app, tasks, TTL beat job
│   │   └── benchmarking.py       # WIDER FACE eval + latency helpers
│   ├── scripts/                  # fetch_weights, benchmark_widerface, export_onnx
│   ├── weights/                  # models live here (git-ignored)
│   └── tests/                    # ~175 tests, no weights needed
├── frontend/                     # React + TS + Tailwind + Vite + Vitest
│   └── src/{api,types}.ts, hooks/, components/, *.test.tsx
├── docs/
│   ├── architecture.md
│   └── benchmarks/               # generated reports (git-ignored)
└── .github/workflows/ci.yml      # backend + frontend + docker jobs
```

---

## Privacy / data retention

- **No** face recognition, identity matching, or embedding storage anywhere in
  the codebase.
- Frames exist only in worker memory for the duration of a job.
- The input file is deleted as soon as a job completes (`DELETE_INPUT_ON_SUCCESS`);
  everything under `/data/<job_id>/` is deleted once `manifest.expires_at` passes
  (`RESULT_TTL_SECONDS`, default 1 h) by the `beat` sweep.
- The worker makes no outbound network calls during processing.

---

## Benchmark results

> Run the scripts (see step 5 above) and paste the generated tables here. Both
> scripts write to `docs/benchmarks/` (git-ignored — numbers are hardware- and
> dataset-specific).

### WIDER FACE (validation) — detector comparison

_From `scripts/benchmark_widerface.py`. Metric is a reproducible greedy-IoU
single pass, **not** the official WIDER Easy/Medium/Hard AP — the report says so._

| Detector | Backbone | Precision @0.5 | Recall @0.5 | AP | Mean FPS | Median latency (ms) |
|---|---|---|---|---|---|---|
| RetinaFace | MobileNet0.25 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| RetinaFace | ResNet50 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| YOLOv8-face | n | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Recall by face size (`max(w,h)` px):

| Detector | small (<32) | medium (32–96) | large (>96) |
|---|---|---|---|
| RetinaFace (MobileNet0.25) | _TBD_ | _TBD_ | _TBD_ |
| YOLOv8-face (n) | _TBD_ | _TBD_ | _TBD_ |

_Hardware: TBD · Command: `python scripts/benchmark_widerface.py --data-root <path>`_

### PyTorch vs ONNX Runtime — YOLOv8-face latency

_From `scripts/export_onnx.py`._

| Runtime | Mean (ms) | p50 (ms) | p95 (ms) | FPS | Speedup |
|---|---|---|---|---|---|
| PyTorch (YOLOv8-face) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | 1.00× |
| ONNX Runtime | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

_Hardware: TBD · Command: `python scripts/export_onnx.py`_

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Every job → `failed`, error mentions "weights not found" | No detector weights in `backend/weights/`. Run `scripts/fetch_weights.py`, or switch `DETECTOR_BACKEND` to the one you have. |
| `docker compose up` → port already in use | Something's on 3000 / 8000 / 6379. Stop it or remap the `ports:` in `docker-compose.yml`. |
| RetinaFace boxes look off / miss obvious faces | The vendored decode is unverified against real weights — use `DETECTOR_BACKEND=yolov8face` and open an issue. |
| `DEVICE=cuda` but it runs on CPU | The image ships CPU torch. Rebuild with `--build-arg BUILD_TORCH_VARIANT=cu121` and use `docker-compose.gpu.yml`. |
| Video output has no audio | The `ffmpeg` audio remux failed (logged as `video.audio_mux_failed`); the video-only result is still produced. |
| Benchmark script exits with code 2 | WIDER FACE not found at `--data-root`. It's not auto-downloaded — see step 5. |
| Frontend loads but uploads hang | The API/worker isn't reachable. Check `docker compose ps` and `http://localhost:8000/healthz`. |

---

## License / intended use

Built for redacting bystanders in footage you have the right to share. **Not**
for surveillance, identification, or tracking individuals. The vendored
RetinaFace code under `backend/app/detectors/_retinaface/` is MIT-licensed
(see the `NOTICE` and `LICENSE` files there).
