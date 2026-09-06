# Scripts

| Script | Deliverable | Purpose |
|---|---|---|
| `benchmark_widerface.py` | 6 | Run both detectors over the WIDER FACE validation set; write precision / recall / inference-fps per backend to `docs/benchmarks/widerface.md`. Expects the dataset at `--data-root` (not auto-downloaded — prints instructions if missing). |
| `export_onnx.py` | 7 | Export **YOLOv8-face** to ONNX (`ultralytics`), then time PyTorch vs ONNX Runtime on the same frames + a correctness sanity check, write `docs/benchmarks/onnx-latency.md`. `--backend retinaface` is not wired (says so). |
| `fetch_weights.py` | 2 | Download RetinaFace + YOLOv8-face weights into `backend/weights/`. |

Run from `backend/`:

```bash
# needs weights (scripts/fetch_weights.py) + the WIDER FACE val set:
#   <root>/WIDER_val/images/...  and  <root>/wider_face_split/wider_face_val_bbx_gt.txt
uv run python scripts/benchmark_widerface.py --data-root ../datasets/widerface
uv run python scripts/benchmark_widerface.py --data-root ../datasets/widerface \
    --backends retinaface --limit 200 --device cpu     # quick single-backend run

uv run python scripts/export_onnx.py --backend yolov8face
```

`benchmark_widerface.py` writes `docs/benchmarks/widerface.md` (git-ignored — numbers
are hardware/dataset specific). Metric is a reproducible greedy-IoU pass, **not** the
official WIDER Easy/Medium/Hard AP; the report states this.
