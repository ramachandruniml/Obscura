"""YOLOv8-face detector (Ultralytics backend).

Speed-oriented backend. Works with plain detection weights and with pose-style
weights that carry 5 facial keypoints; ``provides_landmarks`` reflects which.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.config import settings
from app.detectors.base import Detection, Detector
from app.logging import get_logger

log = get_logger(__name__)


class YOLOv8FaceDetector(Detector):
    name = "yolov8face"
    provides_landmarks = False  # overridden per-instance after inspecting the weights

    def __init__(
        self,
        *,
        weights_path: str | Path | None = None,
        input_size: int | None = None,
        default_confidence: float | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(default_confidence=default_confidence, device=device)
        self.weights_path = Path(weights_path or settings.yolov8_face_weights)
        if not self.weights_path.is_absolute():
            self.weights_path = (Path.cwd() / self.weights_path).resolve()
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"YOLOv8-face weights not found at {self.weights_path}. "
                "Run `python scripts/fetch_weights.py` or set YOLOV8_FACE_WEIGHTS."
            )
        self.input_size = int(input_size or settings.yolov8_input_size)

        # Imported lazily: ultralytics pulls in a heavy dependency tree.
        from ultralytics import YOLO

        self.model = YOLO(str(self.weights_path))
        self.provides_landmarks = getattr(self.model, "task", None) == "pose"

        if self.device == "cuda":
            import torch

            self._device_arg: str | int = 0 if torch.cuda.is_available() else "cpu"
        else:
            self._device_arg = "cpu"

        log.info(
            "detector.loaded",
            detector=self.name,
            device=str(self._device_arg),
            input_size=self.input_size,
            landmarks=self.provides_landmarks,
        )

    def warmup(self) -> None:
        self._detect(np.zeros((64, 64, 3), dtype=np.uint8), self.default_confidence)

    def _detect(self, frame: np.ndarray, confidence: float) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=confidence,
            iou=settings.nms_iou,
            imgsz=self.input_size,
            device=self._device_arg,
            verbose=False,
        )
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        kpts = None
        if self.provides_landmarks and getattr(r, "keypoints", None) is not None:
            kpts = r.keypoints.xy.cpu().numpy()  # (N, K, 2)

        h, w = frame.shape[:2]
        out: list[Detection] = []
        for i, ((x1, y1, x2, y2), s) in enumerate(zip(xyxy, scores, strict=True)):
            lm = None
            if kpts is not None and kpts.shape[1] >= 5:
                lm = kpts[i, :5, :].astype(np.float32)
            out.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=float(s),
                    landmarks=lm,
                ).clip(w, h)
            )
        return out
