"""WIDER FACE evaluation helpers: GT parsing, IoU matching, AP, reporting.

Not the official MATLAB Easy/Medium/Hard protocol (that needs per-image ignore
lists). This is a reproducible greedy-IoU single-pass evaluation, run identically
for every detector so the numbers are comparable to each other. See
``render_report`` for the caveat text that ships in the output.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from app.detectors.base import Detection, Detector
from app.logging import get_logger

log = get_logger(__name__)

SIZE_BUCKETS = ("small", "medium", "large")  # max(w,h): <32, 32-96, >96


@dataclass(frozen=True, slots=True)
class GtBox:
    x1: float
    y1: float
    x2: float
    y2: float
    invalid: bool = False

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_detection(self) -> Detection:
        return Detection(self.x1, self.y1, self.x2, self.y2, 1.0)


def size_bucket(box: GtBox) -> str:
    longest = max(box.width, box.height)
    if longest < 32:
        return "small"
    if longest <= 96:
        return "medium"
    return "large"


def parse_wider_gt(text: str) -> dict[str, list[GtBox]]:
    """Parse ``wider_face_val_bbx_gt.txt`` into ``{relative_image_path: [GtBox, ...]}``.

    Block format: a path line, a face-count line, then that many
    ``x y w h blur expr illum invalid occ pose`` lines. A count of 0 is still
    followed by one all-zero filler line in the official file.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() != ""]
    out: dict[str, list[GtBox]] = {}
    i = 0
    n = len(lines)
    while i < n:
        path = lines[i]
        i += 1
        if i >= n:
            break
        try:
            count = int(lines[i])
        except ValueError:
            # malformed; treat the rest of this token run as no faces
            out[path] = []
            continue
        i += 1
        boxes: list[GtBox] = []
        rows = max(count, 1) if count == 0 else count
        for _ in range(rows):
            if i >= n:
                break
            parts = lines[i].split()
            i += 1
            if count == 0:
                continue
            if len(parts) < 4:
                continue
            x, y, w, h = (float(v) for v in parts[:4])
            invalid = len(parts) >= 8 and parts[7] == "1"
            if w <= 0 or h <= 0:
                continue
            boxes.append(GtBox(x, y, x + w, y + h, invalid=invalid))
        out[path] = boxes
    return out


def _iou(a: Detection | GtBox, b: Detection | GtBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass(slots=True)
class ImageMatch:
    scored: list[tuple[float, bool]] = field(default_factory=list)  # (score, is_tp)
    gt_hits: list[tuple[str, bool]] = field(default_factory=list)  # (size_bucket, matched)
    n_positives: int = 0  # valid (non-invalid) GT boxes


def match_image(dets: Sequence[Detection], gts: Sequence[GtBox], iou_thr: float) -> ImageMatch:
    """Greedy score-ordered matching with WIDER 'invalid' boxes treated as don't-care."""
    valid = [g for g in gts if not g.invalid]
    ignore = [g for g in gts if g.invalid]
    used = [False] * len(valid)

    result = ImageMatch(n_positives=len(valid))
    for det in sorted(dets, key=lambda d: d.score, reverse=True):
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(valid):
            if used[j]:
                continue
            v = _iou(det, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            used[best_j] = True
            result.scored.append((det.score, True))
        elif any(_iou(det, g) >= iou_thr for g in ignore):
            continue  # overlaps a don't-care region: neither TP nor FP
        else:
            result.scored.append((det.score, False))

    for g, hit in zip(valid, used, strict=True):
        result.gt_hits.append((size_bucket(g), hit))
    return result


def average_precision(scored: Iterable[tuple[float, bool]], n_positives: int) -> float:
    """All-point (VOC2010+/COCO-style) area under the precision-recall curve."""
    if n_positives == 0:
        return 0.0
    ordered = sorted(scored, key=lambda s: s[0], reverse=True)
    if not ordered:
        return 0.0
    tp = fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for _, is_tp in ordered:
        tp += int(is_tp)
        fp += int(not is_tp)
        recalls.append(tp / n_positives)
        precisions.append(tp / (tp + fp))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    prev_r = 0.0
    for r, p in zip(recalls, precisions, strict=True):
        ap += (r - prev_r) * p
        prev_r = r
    return ap


def precision_recall_at(
    scored: Sequence[tuple[float, bool]], n_positives: int, score_thr: float
) -> tuple[float, float]:
    tp = sum(1 for s, is_tp in scored if is_tp and s >= score_thr)
    fp = sum(1 for s, is_tp in scored if not is_tp and s >= score_thr)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_positives if n_positives else 0.0
    return precision, recall


@dataclass(slots=True)
class BenchmarkResult:
    detector: str
    n_images: int
    n_faces: int
    operating_threshold: float
    iou_threshold: float
    precision: float
    recall: float
    average_precision: float
    mean_fps: float
    median_latency_ms: float
    recall_by_size: dict[str, float]
    extra: dict[str, str] = field(default_factory=dict)


def _load_bgr(path: Path) -> np.ndarray | None:
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def evaluate_detector(
    detector: Detector,
    samples: Sequence[tuple[Path, list[GtBox]]],
    *,
    iou_thr: float = 0.5,
    min_score: float = 0.01,
    operating_threshold: float | None = None,
    warmup: int = 10,
    progress_every: int = 200,
) -> BenchmarkResult:
    op_thr = detector.default_confidence if operating_threshold is None else operating_threshold

    for path, _ in samples[: max(0, warmup)]:
        frame = _load_bgr(path)
        if frame is not None:
            detector.detect(frame, min_score)

    all_scored: list[tuple[float, bool]] = []
    total_positives = 0
    hit_by_bucket: Counter[str] = Counter()
    tot_by_bucket: Counter[str] = Counter()
    latencies: list[float] = []
    n_faces = 0

    for idx, (path, gts) in enumerate(samples):
        frame = _load_bgr(path)
        if frame is None:
            log.warning("benchmark.unreadable", path=str(path))
            continue
        t0 = time.perf_counter()
        dets = detector.detect(frame, min_score)
        latencies.append(time.perf_counter() - t0)

        m = match_image(dets, gts, iou_thr)
        all_scored.extend(m.scored)
        total_positives += m.n_positives
        n_faces += m.n_positives
        for bucket, matched in m.gt_hits:
            tot_by_bucket[bucket] += 1
            if matched:
                hit_by_bucket[bucket] += 1

        if progress_every and (idx + 1) % progress_every == 0:
            log.info("benchmark.progress", detector=detector.name, images=idx + 1, of=len(samples))

    precision, recall = precision_recall_at(all_scored, total_positives, op_thr)
    ap = average_precision(all_scored, total_positives)
    total_time = sum(latencies) or 1e-9
    recall_by_size = {
        b: (hit_by_bucket[b] / tot_by_bucket[b] if tot_by_bucket[b] else 0.0) for b in SIZE_BUCKETS
    }

    return BenchmarkResult(
        detector=detector.name,
        n_images=len(latencies),
        n_faces=n_faces,
        operating_threshold=op_thr,
        iou_threshold=iou_thr,
        precision=precision,
        recall=recall,
        average_precision=ap,
        mean_fps=len(latencies) / total_time,
        median_latency_ms=float(np.median(latencies) * 1000) if latencies else 0.0,
        recall_by_size=recall_by_size,
    )


def render_report(results: Sequence[BenchmarkResult], meta: dict[str, str]) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# WIDER FACE — detector comparison",
        "",
        f"_Generated {now}_",
        "",
        "## Run",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for k, v in meta.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Results",
        "",
        "| Detector | Precision | Recall | AP | Mean FPS | Median latency (ms) | Images | Faces |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.detector} | {r.precision:.3f} | {r.recall:.3f} | {r.average_precision:.3f} "
            f"| {r.mean_fps:.1f} | {r.median_latency_ms:.1f} | {r.n_images} | {r.n_faces} |"
        )

    lines += ["", "### Recall by face size (`max(w, h)` px)", "",
              "| Detector | small (<32) | medium (32–96) | large (>96) |",
              "|---|---|---|---|"]  # fmt: skip
    for r in results:
        s = r.recall_by_size
        lines.append(f"| {r.detector} | {s['small']:.3f} | {s['medium']:.3f} | {s['large']:.3f} |")

    op = results[0].operating_threshold if results else 0.0
    iou = results[0].iou_threshold if results else 0.5
    lines += [
        "",
        "## Notes",
        "",
        f"- Precision/Recall reported at operating score threshold **{op:.2f}**; "
        f"AP is the all-point area under the PR curve (detections scored down to 0.01).",
        f"- A detection is a true positive at **IoU ≥ {iou:.2f}** against an unclaimed "
        "ground-truth box. WIDER `invalid` boxes are treated as don't-care.",
        "- **This is not the official WIDER Easy/Medium/Hard AP protocol** (which uses "
        "per-image ignore lists and curve settings from the devkit). Numbers here are "
        "for comparing these two detectors on identical inputs, not for leaderboard parity.",
        "- FPS is wall-clock around `detector.detect()` only (image decode excluded), "
        "after a warmup pass.",
        "",
    ]
    cmd = meta.get("command", "python scripts/benchmark_widerface.py --data-root <path>")
    lines.append(f"Command: `{cmd}`")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Latency measurement (PyTorch vs ONNX Runtime)                                #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class LatencyStats:
    label: str
    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    fps: float


def measure_latency(
    fn: Callable[[object], object], inputs: Sequence, *, warmup: int = 5, label: str = ""
) -> LatencyStats:
    """Time ``fn(x)`` for each x in ``inputs`` (after ``warmup`` untimed calls)."""
    for x in inputs[: max(0, warmup)]:
        fn(x)
    times: list[float] = []
    for x in inputs:
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times) if times else np.zeros(1)
    mean = float(arr.mean())
    return LatencyStats(
        label=label,
        n=len(times),
        mean_ms=mean,
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        fps=(1000.0 / mean) if mean > 0 else 0.0,
    )


def render_latency_report(
    baseline: LatencyStats, optimized: LatencyStats, meta: dict[str, str]
) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    speedup = baseline.mean_ms / optimized.mean_ms if optimized.mean_ms > 0 else 0.0
    lines = [
        "# PyTorch vs ONNX Runtime — inference latency",
        "",
        f"_Generated {now}_",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in meta.items()]
    lines += [
        "",
        "| Runtime | Mean (ms) | p50 (ms) | p95 (ms) | FPS | Speedup |",
        "|---|---|---|---|---|---|",
        f"| {baseline.label} | {baseline.mean_ms:.2f} | {baseline.p50_ms:.2f} "
        f"| {baseline.p95_ms:.2f} | {baseline.fps:.1f} | 1.00x |",
        f"| {optimized.label} | {optimized.mean_ms:.2f} | {optimized.p50_ms:.2f} "
        f"| {optimized.p95_ms:.2f} | {optimized.fps:.1f} | {speedup:.2f}x |",
        "",
        "## Notes",
        "",
        f"- {baseline.n} timed inferences each, after a warmup pass; timing wraps "
        "`detector.detect()` (letterbox + forward + decode + NMS), decode of the "
        "source image excluded.",
        "- Same frames fed to both runtimes.",
        "",
    ]
    cmd = meta.get("command", "python scripts/export_onnx.py")
    lines.append(f"Command: `{cmd}`")
    lines.append("")
    return "\n".join(lines)
