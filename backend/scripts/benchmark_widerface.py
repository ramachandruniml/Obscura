"""Benchmark RetinaFace and YOLOv8-face on the WIDER FACE validation set.

Run from backend/:

    uv run python scripts/benchmark_widerface.py --data-root ../datasets/widerface

Expected layout under --data-root:

    WIDER_val/images/<event>/<name>.jpg
    wider_face_split/wider_face_val_bbx_gt.txt

The dataset is NOT downloaded automatically. Get it from:
  - http://shuoyang1213.me/WIDERFACE/  (WIDER_val.zip + wider_face_split.zip)
  - or `huggingface-cli download wider_face --repo-type dataset`

Writes a markdown report to --out (default docs/benchmarks/widerface.md).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.benchmarking import (  # noqa: E402
    BenchmarkResult,
    GtBox,
    evaluate_detector,
    parse_wider_gt,
    render_report,
)
from app.config import settings  # noqa: E402
from app.detectors.registry import get_detector_class  # noqa: E402
from app.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("benchmark")

DEFAULT_OUT = REPO_ROOT.parent / "docs" / "benchmarks" / "widerface.md"


def _find_layout(root: Path) -> tuple[Path, Path]:
    gt = next(root.rglob("wider_face_val_bbx_gt.txt"), None)
    images = root / "WIDER_val" / "images"
    if not images.is_dir():
        cand = next((p for p in root.rglob("images") if p.is_dir()), None)
        images = cand or images
    if gt is None or not images.is_dir():
        raise FileNotFoundError(
            "WIDER FACE not found under "
            f"{root}.\nExpected WIDER_val/images/ and wider_face_split/wider_face_val_bbx_gt.txt.\n"
            "Download: http://shuoyang1213.me/WIDERFACE/  (WIDER_val.zip + wider_face_split.zip)"
        )
    return images, gt


def _load_samples(
    images_dir: Path, gt_path: Path, limit: int | None
) -> list[tuple[Path, list[GtBox]]]:
    gt = parse_wider_gt(gt_path.read_text(encoding="utf-8"))
    samples: list[tuple[Path, list[GtBox]]] = []
    for rel, boxes in sorted(gt.items()):
        p = images_dir / rel
        if p.exists():
            samples.append((p, boxes))
        if limit and len(samples) >= limit:
            break
    if not samples:
        raise FileNotFoundError(f"no images matched GT entries under {images_dir}")
    return samples


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--backends", default="retinaface,yolov8face")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-score", type=float, default=0.01)
    ap.add_argument("--operating-threshold", type=float, default=settings.default_confidence)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N images")
    ap.add_argument("--device", default=settings.device, choices=["cpu", "cuda"])
    args = ap.parse_args(argv)

    configure_logging()
    try:
        images_dir, gt_path = _find_layout(args.data_root)
    except FileNotFoundError as exc:
        log.error("benchmark.dataset_missing", detail=str(exc))
        print(exc, file=sys.stderr)
        return 2

    samples = _load_samples(images_dir, gt_path, args.limit)
    log.info("benchmark.loaded", images=len(samples), gt=str(gt_path))

    results: list[BenchmarkResult] = []
    for name in [b.strip() for b in args.backends.split(",") if b.strip()]:
        try:
            cls = get_detector_class(name)
        except ValueError as exc:
            log.warning("benchmark.skip_backend", backend=name, reason=str(exc))
            continue
        try:
            detector = cls(device=args.device)
        except FileNotFoundError as exc:
            log.warning("benchmark.skip_backend", backend=name, reason=str(exc))
            continue

        log.info("benchmark.start", backend=name, device=args.device)
        results.append(
            evaluate_detector(
                detector,
                samples,
                iou_thr=args.iou,
                min_score=args.min_score,
                operating_threshold=args.operating_threshold,
                warmup=args.warmup,
            )
        )

    if not results:
        print("no detectors ran (missing weights?). See scripts/fetch_weights.py", file=sys.stderr)
        return 1

    meta = {
        "dataset": str(args.data_root),
        "images evaluated": str(len(samples)),
        "device": args.device,
        "git sha": _git_sha(),
        "IoU threshold": f"{args.iou:.2f}",
        "operating threshold": f"{args.operating_threshold:.2f}",
        "command": "python "
        + " ".join(["scripts/benchmark_widerface.py", *(argv or sys.argv[1:])]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(results, meta), encoding="utf-8")
    log.info("benchmark.done", out=str(args.out), backends=[r.detector for r in results])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
