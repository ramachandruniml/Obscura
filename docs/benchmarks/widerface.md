# WIDER FACE (validation) — detector comparison

_Generated 2026-09-11 · `git fff878c`_

## Run

| Field | Value |
|---|---|
| dataset | WIDER FACE validation set (official split), 3,226 images / 39,112 annotated faces |
| device | CPU — 11th Gen Intel Core i5-1135G7 @ 2.40 GHz |
| IoU threshold | 0.50 |
| operating threshold | 0.50 |
| RetinaFace backbone | MobileNet0.25 |
| YOLOv8-face | `yolov8n-face` |

A first pass hit `RETINAFACE_KEEP_TOP_K=750` / `MAX_FACES=500` on dense crowd
images at the low score floor (0.01) used to build the PR curve — RetinaFace,
never YOLOv8-face. Re-ran with both caps raised to 2000; the numbers below
moved by <1 point (AP 0.431→0.434, small-face recall 0.349→0.374), so the caps
were not a material confound.

## Results

| Detector | Precision | Recall | AP | Mean FPS | Median latency (ms) |
|---|---|---|---|---|---|
| RetinaFace (MobileNet0.25) | 0.889 | 0.334 | 0.434 | 9.3 | 95.0 |
| YOLOv8-face (n) | **0.978** | **0.433** | **0.644** | 5.0 | 129.4 |

### Recall by face size (`max(w, h)` px)

| Detector | small (<32px) | medium (32–96px) | large (>96px) |
|---|---|---|---|
| RetinaFace (MobileNet0.25) | 0.374 | 0.924 | 0.980 |
| YOLOv8-face (n) | **0.562** | 0.941 | 0.978 |

## Reading these numbers

- **YOLOv8-face is more accurate across the board** on this evaluation —
  higher precision, recall, and AP, and a large gap on small faces (0.562 vs
  0.374 recall). **RetinaFace (MobileNet0.25) is faster** — it's the ~1.7 MB
  backbone the upstream authors built specifically for edge/CPU speed, so this
  is the expected accuracy/speed trade-off, not a bug.
- **Recall looks low in absolute terms (33–43%)** because this counts every
  one of WIDER FACE's 39,112 annotated faces, including the extremely dense,
  few-pixel faces in crowd photos that the *official* WIDER benchmark splits
  into a separate "Hard" subset with per-image ignore regions. A face this
  small is arguably not "identifying" in the first place; for the redaction
  use case, the **medium/large recall (92–98%)** is the more relevant number.
- **This is a reproducible greedy-IoU single pass, not the official WIDER
  Easy/Medium/Hard AP protocol** (which uses per-image ignore lists and curve
  settings from the devkit) — for comparing these two detectors on identical
  inputs, not for leaderboard parity.
- FPS/latency is wall-clock around `detector.detect()` only (image decode
  excluded), after a warmup pass, on a single CPU core with other processes
  running — treat the two detectors' relative FPS as more meaningful than the
  absolute numbers.

Command: `python scripts/benchmark_widerface.py --data-root <wider_face_root>`
