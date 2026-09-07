# Architecture

Obscura is a job queue with a computer-vision worker. A browser uploads a file,
the API validates it and enqueues a job, a Celery worker runs the
detect → track → redact pipeline, and the browser polls until the redacted file
is ready. Nothing is kept: the input is deleted on success, everything else on a
TTL.

## Services (Docker Compose)

| Service | Image / build | Role |
|---|---|---|
| `redis` | `redis:7-alpine` | Celery broker (db 0) + result backend (db 1). `--save "" --appendonly no` — nothing to persist. |
| `api` | `backend/` | FastAPI. Validates uploads, streams them to the shared volume, writes `manifest.json`, enqueues a Celery task, serves status + results. **Never runs the model in-process.** |
| `worker` | `backend/` (same image) | Celery worker. Runs the redaction pipeline. `--concurrency=1` — CPU inference is the bottleneck; parallelism comes from running more worker containers. |
| `beat` | `backend/` (same image) | Celery beat. Fires `sweep_expired_artifacts` every `CLEANUP_INTERVAL_SECONDS`. |
| `frontend` | `frontend/` | React build served by nginx; `location /api/` proxies to `api:8000`, so the browser only ever talks to one origin. |

One named volume, `obscura-data`, mounted at `/data` in `api`, `worker`, and
`beat`. `backend/weights/` is bind-mounted into `api` and `worker` so weights you
drop on the host appear in the containers.

## Job lifecycle

```
POST /api/redact  (multipart: file, method, confidence)
  api:
    media.classify(content_type, filename)         -> 415 on unknown type
    media.stream_to_file(..., max=MAX_UPLOAD_BYTES) -> 413 while streaming
    media.probe(path, kind)                          -> 422 corrupt / 422 too long
    storage.create_job_dir(uuid)
    storage.init_manifest(status=pending, kind, method, confidence, expires_at)
    run_redaction_job.delay(job_id)
    -> 202 {job_id, status: "pending"}

worker (run_redaction_job -> pipeline.runner.process_job):
    manifest.status = "processing"
    detector  = get_cached_detector()      # loaded once per process, then reused
    redactor  = build_redactor(manifest.method)
    if kind == image:  redact_image(src, dst, detector, redactor, confidence)
    else:              redact_video(src, dst, detector, redactor, confidence)
    manifest.status = "complete" | "failed"  (+ stats | + error)
    on success: storage.delete_input(job_id)

browser:
    poll GET /api/jobs/<id> every 1.5 s until status in {complete, failed}
    GET /api/jobs/<id>/result   -> FileResponse

beat (every CLEANUP_INTERVAL_SECONDS):
    storage.sweep(): rm -rf every /data/<job>/ whose manifest.expires_at < now
                     (falls back to directory mtime if the manifest is unreadable)
```

`manifest.json` on the shared volume is the **single source of truth** for job
status — the API writes it, the worker updates it (atomic temp-file + `os.replace`
so the two containers never read a torn file), the sweep consumes its
`expires_at`. Celery's Redis result backend still records task execution but
nothing user-facing depends on it.

## Detection pipeline

### Image (`app/pipeline/image_pipeline.py`)

`cv2.imdecode` → `detector.detect(frame, confidence)` → `redactor.apply(frame, dets)`
→ `cv2.imencode` (keeps the input format). A `None` decode raises
`CorruptedMediaError`.

### Video (`app/pipeline/video_pipeline.py`)

PyAV demuxes the source. For each frame:

- **every Nth frame** (`DETECT_EVERY_N_FRAMES`): run the detector, feed the
  detections to `FaceTracker.update(idx, dets)` → ByteTrack assigns/refreshes
  stable `track_id`s.
- **in between**: `FaceTracker.update(idx, None)` → each live track's box is
  extrapolated from its last two real observations (constant velocity, size
  held), clipped to the frame, flagged `coasting`. This is what stops the
  redaction flickering on the ~80% of frames where no detector runs.

A coasted track is dropped when its predicted box leaves the frame or after
`3 × DETECT_EVERY_N_FRAMES` un-confirmed frames. A face re-entering after a long
gap gets a **new** id — no re-identification, deliberately (that would be the
recognition this project avoids).

Each frame is redacted in place, encoded with libx264 (`yuv420p`, always `.mp4`
out so the browser can preview it), then the source audio is remuxed back with
`ffmpeg -c copy` (via `imageio-ffmpeg`'s bundled binary). If the source has no
audio, or the mux fails, the video-only result is used.

## Detector interface (`app/detectors/`)

```
Detection(x1, y1, x2, y2, score, landmarks?)     # xyxy in absolute pixels
Detector.detect(frame, confidence) -> list[Detection]
    validate frame (HxWx3 uint8) -> _detect(...) -> filter by score,
    sort desc, cap at MAX_FACES
```

| Implementation | Source | Notes |
|---|---|---|
| `RetinaFaceDetector` | vendored [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) under `_retinaface/` | MobileNet0.25 / ResNet50 backbones, FPN + SSH, 5-pt landmarks. NMS via `torchvision.ops.nms`. Accuracy-oriented, strong on small / angled faces. |
| `YOLOv8FaceDetector` | `ultralytics` | Thin wrapper. `provides_landmarks` auto-detected from the weights (pose vs plain detection). Speed-oriented. |
| `OnnxFaceDetector` | `onnxruntime` + `_yolo_onnx.py` | Consumes a raw YOLOv8-face ONNX export; letterbox + decode + NMS in numpy, no torch/ultralytics at inference. Enabled by `DETECTOR_RUNTIME=onnx`. |

`app/detectors/registry.py` maps `DETECTOR_BACKEND` (+ `DETECTOR_RUNTIME`) to an
instance and caches it per process.

## Redaction (`app/redaction/`)

`Redactor.apply(frame, regions, pad_ratio=, copy=)` — expands every box by
`BOX_PADDING_RATIO` (via `Detection.expand`, then clips to the frame), skips
degenerate / off-frame boxes, applies the effect to the ROI:

- **GaussianBlur** — kernel scales with the box's short side
  (`BLUR_KERNEL_FRACTION`, floor `BLUR_MIN_KERNEL`), `BLUR_PASSES` passes for
  irreversibility.
- **Pixelate** — downscale to `~PIXELATE_BLOCKS` cells (aspect kept),
  `INTER_NEAREST` upscale → a hard block grid.
- **SolidBox** — fill with `SOLID_BOX_COLOR` (BGR).

The video pipeline calls `apply(copy=False)` on its own frame buffer; the image
pipeline copies.

## Observability

`app/logging.py` (structlog) binds `job_id` to the context for the life of a job.
Grep one job end to end:

```
upload.received job_id=… kind=image method=blur bytes=…
detection.started job_id=… detector=retinaface width=… height=…
detection.completed job_id=… n_faces=3
tracking.completed job_id=… n_tracks=4 n_detection_frames=24 frames=118   (video)
redaction.applied job_id=… method=blur n_regions=3
job.completed job_id=… kind=image n_faces=3
```

Set `LOG_JSON=true` for machine-readable logs.

## Privacy properties (enforced by construction)

- No recognition / embedding / matching code exists anywhere in `backend/app/`.
- Frames are local variables in the worker; never written except as the encoded
  output.
- Input deleted on success; the whole job directory deleted at `expires_at`.
- The worker opens no sockets during processing (model weights are loaded from
  disk at startup).
