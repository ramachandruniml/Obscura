"""ONNX Runtime face detector (YOLOv8-face graph).

Used when ``DETECTOR_RUNTIME=onnx``. Consumes a raw YOLOv8(-face) ONNX export
(see scripts/export_onnx.py); does letterbox + decode + NMS in numpy. Depends
only on onnxruntime + numpy + cv2 at inference time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.config import settings
from app.detectors._yolo_onnx import (
    decode_yolo_output,
    keypoint_count,
    letterbox,
    to_blob,
)
from app.detectors.base import Detection, Detector
from app.logging import get_logger

log = get_logger(__name__)


class OnnxFaceDetector(Detector):
    name = "onnx"
    provides_landmarks = False  # set per-instance from the graph outputs

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        input_size: int | None = None,
        default_confidence: float | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(default_confidence=default_confidence, device=device)
        self.model_path = Path(model_path or settings.onnx_model_path)
        if not self.model_path.is_absolute():
            self.model_path = (Path.cwd() / self.model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {self.model_path}. Run "
                "`python scripts/export_onnx.py` or set ONNX_MODEL_PATH."
            )
        self.input_size = int(input_size or settings.yolov8_input_size)

        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # infer keypoint count from a dummy forward
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        out = self.session.run(None, {self.input_name: dummy})[0]
        channels = out.shape[1] if out.ndim == 3 and out.shape[1] < out.shape[2] else out.shape[-1]
        self.n_kpt = keypoint_count(int(channels))
        self.provides_landmarks = self.n_kpt >= 5

        log.info(
            "detector.loaded",
            detector=self.name,
            model=str(self.model_path),
            providers=providers,
            input_size=self.input_size,
            n_kpt=self.n_kpt,
        )

    def warmup(self) -> None:
        self._detect(np.zeros((64, 64, 3), dtype=np.uint8), self.default_confidence)

    def _detect(self, frame: np.ndarray, confidence: float) -> list[Detection]:
        lb, ratio, pad = letterbox(frame, self.input_size)
        raw = self.session.run(None, {self.input_name: to_blob(lb)})[0]
        return decode_yolo_output(
            raw,
            ratio=ratio,
            pad=pad,
            orig_shape=frame.shape[:2],
            conf=confidence,
            iou=settings.nms_iou,
            n_kpt=self.n_kpt,
            max_det=settings.max_faces,
        )
