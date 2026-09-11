# PyTorch vs ONNX Runtime — inference latency

_Generated 2026-09-11 00:40 UTC · `git 0e8b391`_

| Field | Value |
|---|---|
| model | YOLOv8-face (yolov8n-face.pt, 3.08M params, 8.3 GFLOPs) |
| onnx | `weights/detector.onnx` (opset 12, 640×640) |
| device | CPU — 11th Gen Intel Core i5-1135G7 @ 2.40 GHz |
| frames | 100 synthetic, warmup 12 |
| sanity: max Δ(detection count), PyTorch vs ONNX | 0 |

| Runtime | Mean (ms) | p50 (ms) | p95 (ms) | FPS | Speedup |
|---|---|---|---|---|---|
| PyTorch (YOLOv8-face) | 185.53 | 166.63 | 265.04 | 5.4 | 1.00× |
| ONNX Runtime | 99.91 | 100.15 | 108.04 | 10.0 | **1.86×** |

## Notes

- 100 timed inferences each, after a warmup pass; timing wraps
  `detector.detect()` — letterbox + forward + decode + NMS — with source-image
  decode excluded. Same frames fed to both runtimes.
- ONNX Runtime is not just faster on average, it's far more consistent:
  p95 108 ms vs 265 ms.
- The ONNX consumer runs a raw YOLOv8 export (no bundled NMS) and reimplements
  letterbox + box decode + NMS in NumPy, so the inference path needs only
  `onnxruntime` + `numpy` + `opencv` (no PyTorch, no Ultralytics).

Command: `python scripts/export_onnx.py --runs 100 --warmup 12`
