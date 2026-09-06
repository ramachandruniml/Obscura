"""Pure pre/post-processing for a raw YOLOv8(-face) ONNX graph.

No onnxruntime / torch / ultralytics here — just numpy + cv2 — so the decode
path is unit-testable without a model. ``OnnxFaceDetector`` wires these to an
InferenceSession.

Raw YOLOv8 export output is ``(1, C, A)`` (or ``(1, A, C)``): C = 4 box
(cx, cy, w, h in letterboxed pixels) + 1 class score [+ 3*K keypoints], A anchors.
Boxes and keypoints are already decoded and sigmoid-activated in the graph.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.detectors.base import Detection

_PAD_COLOR = 114


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to ``size x size``.

    Returns ``(padded_bgr, ratio, (pad_left, pad_top))``. Unmap a letterboxed
    coordinate with ``orig = (lb - pad) / ratio``.
    """
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    nh, nw = round(h * ratio), round(w * ratio)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), _PAD_COLOR, dtype=np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    out[top : top + nh, left : left + nw] = resized
    return out, ratio, (left, top)


def to_blob(letterboxed_bgr: np.ndarray) -> np.ndarray:
    """BGR HWC uint8 -> RGB CHW float32 [0,1] with a batch axis."""
    rgb = letterboxed_bgr[:, :, ::-1]
    blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(blob)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS on xyxy boxes. Returns kept indices, highest score first."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def _orient(raw: np.ndarray, expected_c: int | None) -> np.ndarray:
    """Normalize a raw YOLOv8 export output to ``(num_anchors, channels)``.

    Raw (no-NMS) export is channels-first ``(1, C, A)``. When the channel count
    is known it is used to pick the right axis; otherwise the smaller axis is
    assumed to be the channels (true for real exports, A >> C).
    """
    arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"unexpected YOLO ONNX output shape {arr.shape}")
    if expected_c is not None:
        if arr.shape[1] == expected_c:
            return np.ascontiguousarray(arr)
        if arr.shape[0] == expected_c:
            return np.ascontiguousarray(arr.T)
    return np.ascontiguousarray(arr.T if arr.shape[0] < arr.shape[1] else arr)


def keypoint_count(channels: int) -> int:
    """Number of keypoints implied by the channel count (nc=1 face models)."""
    return (channels - 5) // 3 if channels > 5 and (channels - 5) % 3 == 0 else 0


def decode_yolo_output(
    raw: np.ndarray,
    *,
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
    conf: float,
    iou: float,
    n_kpt: int | None = None,
    max_det: int = 500,
) -> list[Detection]:
    """Raw ONNX output -> list of Detection in original-image pixels."""
    expected_c = None if n_kpt is None else 5 + 3 * n_kpt
    arr = _orient(raw, expected_c)
    if n_kpt is None:
        n_kpt = keypoint_count(arr.shape[1])

    scores = arr[:, 4]
    mask = scores >= conf
    arr, scores = arr[mask], scores[mask]
    if arr.shape[0] == 0:
        return []

    cx, cy, w, h = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    pad_l, pad_t = pad
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_l) / ratio
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_t) / ratio
    oh, ow = orig_shape
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, ow)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, oh)

    kpts = None
    if n_kpt >= 5:
        k = arr[:, 5 : 5 + n_kpt * 3].reshape(-1, n_kpt, 3)[:, :5, :2].astype(np.float32)
        k[:, :, 0] = (k[:, :, 0] - pad_l) / ratio
        k[:, :, 1] = (k[:, :, 1] - pad_t) / ratio
        kpts = k

    keep = nms_numpy(xyxy, scores, iou)[:max_det]
    out: list[Detection] = []
    for i in keep:
        lm = kpts[i] if kpts is not None else None
        out.append(
            Detection(
                float(xyxy[i, 0]),
                float(xyxy[i, 1]),
                float(xyxy[i, 2]),
                float(xyxy[i, 3]),
                float(scores[i]),
                lm,
            )
        )
    return out
