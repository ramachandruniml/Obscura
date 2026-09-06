# Scripts

| Script | Deliverable | Purpose |
|---|---|---|
| `benchmark_widerface.py` | 6 | Run both detectors over the WIDER FACE validation set; write precision / recall / inference-fps per backend to `docs/benchmarks/widerface.md`. Expects the dataset at `--data-root` (not auto-downloaded — prints instructions if missing). |
| `export_onnx.py` | 7 | Export the selected PyTorch detector to ONNX, benchmark PyTorch vs ONNX Runtime latency, write `docs/benchmarks/onnx-latency.md`. |
| `fetch_weights.py` | 2 | Download RetinaFace + YOLOv8-face weights into `backend/weights/`. |

Run from `backend/`:

```bash
uv run python scripts/benchmark_widerface.py --data-root ../datasets/widerface
uv run python scripts/export_onnx.py --backend yolov8face
```
