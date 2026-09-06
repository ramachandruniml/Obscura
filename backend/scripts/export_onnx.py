"""Export YOLOv8-face to ONNX and compare PyTorch vs ONNX Runtime latency.

Run from backend/:

    uv run python scripts/export_onnx.py                       # export + benchmark
    uv run python scripts/export_onnx.py --images ../datasets/widerface/WIDER_val/images
    uv run python scripts/export_onnx.py --no-export --runs 100 # benchmark an existing .onnx

RetinaFace ONNX is not wired: YOLOv8-face is the supported ONNX path (clean
ultralytics export, stable raw output). --backend retinaface just says so.

Writes docs/benchmarks/onnx-latency.md.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.benchmarking import measure_latency, render_latency_report  # noqa: E402
from app.config import settings  # noqa: E402
from app.detectors.base import iou  # noqa: E402
from app.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("export_onnx")
DEFAULT_OUT = REPO_ROOT.parent / "docs" / "benchmarks" / "onnx-latency.md"


def export_yolov8_onnx(weights: Path, dst: Path, imgsz: int, opset: int) -> Path:
    from ultralytics import YOLO

    log.info("onnx.export.start", weights=str(weights), imgsz=imgsz, opset=opset)
    produced = Path(
        YOLO(str(weights)).export(
            format="onnx", imgsz=imgsz, opset=opset, dynamic=False, simplify=False
        )
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != dst.resolve():
        shutil.copy2(produced, dst)
    log.info("onnx.export.done", path=str(dst), bytes=dst.stat().st_size)
    return dst


def load_frames(images_dir: Path | None, count: int, size: int) -> list[np.ndarray]:
    import cv2

    if images_dir and images_dir.is_dir():
        paths = sorted(
            p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        frames = []
        for p in paths[:count]:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
        if frames:
            log.info("onnx.frames", source=str(images_dir), n=len(frames))
            return frames
    rng = np.random.default_rng(0)
    log.info("onnx.frames", source="synthetic", n=count)
    return [rng.integers(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(count)]


def sanity_check(
    pt_detector, onnx_detector, frames: list[np.ndarray], conf: float
) -> dict[str, str]:
    diffs, matched_iou = [], []
    for f in frames[:10]:
        a = pt_detector.detect(f, conf)
        b = onnx_detector.detect(f, conf)
        diffs.append(abs(len(a) - len(b)))
        for da in a:
            best = max((iou(da, db) for db in b), default=0.0)
            if best > 0:
                matched_iou.append(best)
    mean_iou = float(np.mean(matched_iou)) if matched_iou else 0.0
    ok = max(diffs, default=0) <= 2 and (not matched_iou or mean_iou >= 0.7)
    if not ok:
        log.warning(
            "onnx.sanity.divergent", max_count_diff=max(diffs, default=0), mean_iou=mean_iou
        )
    return {
        "sanity: max Δ(detection count)": str(max(diffs, default=0)),
        "sanity: mean IoU of matched boxes": f"{mean_iou:.3f}",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backend", default="yolov8face", choices=["yolov8face", "retinaface"])
    ap.add_argument("--weights", type=Path, default=settings.yolov8_face_weights)
    ap.add_argument("--onnx-out", type=Path, default=settings.onnx_model_path)
    ap.add_argument("--report-out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--imgsz", type=int, default=settings.yolov8_input_size)
    ap.add_argument("--opset", type=int, default=settings.onnx_opset)
    ap.add_argument("--images", type=Path, default=None, help="dir of frames to time on")
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--device", default=settings.device, choices=["cpu", "cuda"])
    ap.add_argument("--no-export", action="store_true", help="benchmark an existing .onnx")
    args = ap.parse_args(argv)

    configure_logging()

    if args.backend == "retinaface":
        print(
            "RetinaFace ONNX export is not wired. YOLOv8-face is the supported ONNX path "
            "(ultralytics export + stable raw output). Re-run with --backend yolov8face.",
            file=sys.stderr,
        )
        return 2

    if not args.no_export:
        if not Path(args.weights).exists():
            print(
                f"weights not found: {args.weights} (run scripts/fetch_weights.py)", file=sys.stderr
            )
            return 1
        export_yolov8_onnx(Path(args.weights), Path(args.onnx_out), args.imgsz, args.opset)
    elif not Path(args.onnx_out).exists():
        print(f"--no-export but {args.onnx_out} is missing", file=sys.stderr)
        return 1

    from app.detectors.onnx_detector import OnnxFaceDetector
    from app.detectors.yolov8_face import YOLOv8FaceDetector

    frames = load_frames(args.images, args.runs, args.imgsz)
    conf = settings.default_confidence

    pt = YOLOv8FaceDetector(device=args.device)
    onnx = OnnxFaceDetector(model_path=args.onnx_out, input_size=args.imgsz, device=args.device)

    meta_extra = sanity_check(pt, onnx, frames, conf)

    baseline = measure_latency(
        lambda f: pt.detect(f, conf), frames, warmup=args.warmup, label="PyTorch (YOLOv8-face)"
    )
    optimized = measure_latency(
        lambda f: onnx.detect(f, conf), frames, warmup=args.warmup, label="ONNX Runtime"
    )

    meta = {
        "model": f"YOLOv8-face ({Path(args.weights).name})",
        "onnx": str(args.onnx_out),
        "device": args.device,
        "input size": str(args.imgsz),
        "frames": f"{len(frames)} ({'real' if args.images else 'synthetic'})",
        **meta_extra,
        "command": "python " + " ".join(["scripts/export_onnx.py", *(argv or sys.argv[1:])]),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_latency_report(baseline, optimized, meta), encoding="utf-8")
    log.info(
        "onnx.done",
        out=str(args.report_out),
        pt_ms=round(baseline.mean_ms, 2),
        onnx_ms=round(optimized.mean_ms, 2),
        speedup=round(baseline.mean_ms / max(optimized.mean_ms, 1e-9), 2),
    )
    print(f"wrote {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
