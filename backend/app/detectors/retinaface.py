"""RetinaFace detector (biubug6 Pytorch_Retinaface backend).

Accuracy-oriented backend: 5-point landmarks, strong on small / angled faces.
Preprocessing follows the upstream ``test_widerface.py`` path (resize long side
to ``settings.retinaface_input_size``, subtract the BGR mean, no [0,1] scaling).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from app.config import settings
from app.detectors._retinaface import (
    BACKBONES,
    PriorBox,
    decode,
    decode_landm,
    load_retinaface,
)
from app.detectors.base import Detection, Detector
from app.logging import get_logger

log = get_logger(__name__)

_BGR_MEAN = (104.0, 117.0, 123.0)


class RetinaFaceDetector(Detector):
    name = "retinaface"
    provides_landmarks = True

    def __init__(
        self,
        *,
        weights_path: str | Path | None = None,
        backbone: str | None = None,
        input_size: int | None = None,
        default_confidence: float | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(default_confidence=default_confidence, device=device)
        self.backbone = backbone or settings.retinaface_backbone
        if self.backbone not in BACKBONES:
            raise ValueError(f"unknown retinaface backbone {self.backbone!r}")
        self.cfg = BACKBONES[self.backbone]
        self.input_size = int(input_size or settings.retinaface_input_size)
        self.max_side = int(settings.retinaface_max_side)

        self.weights_path = Path(weights_path or settings.retinaface_weights)
        if not self.weights_path.is_absolute():
            self.weights_path = (Path.cwd() / self.weights_path).resolve()
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"RetinaFace weights not found at {self.weights_path}. "
                "Run `python scripts/fetch_weights.py` or set RETINAFACE_WEIGHTS."
            )

        self._torch_device = torch.device(
            "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.model = load_retinaface(self.cfg, str(self.weights_path), self._torch_device)
        log.info(
            "detector.loaded",
            detector=self.name,
            backbone=self.backbone,
            device=str(self._torch_device),
            input_size=self.input_size,
        )

    # -- helpers --------------------------------------------------------------
    def _resize_factor(self, h: int, w: int) -> float:
        resize = self.input_size / max(h, w)
        if max(h, w) * resize > self.max_side:
            resize = self.max_side / max(h, w)
        return resize

    # -- Detector API -------------------------------------------------------
    def warmup(self) -> None:
        self._detect(np.zeros((64, 64, 3), dtype=np.uint8), self.default_confidence)

    @torch.no_grad()
    def _detect(self, frame: np.ndarray, confidence: float) -> list[Detection]:
        h0, w0 = frame.shape[:2]
        resize = self._resize_factor(h0, w0)

        if abs(resize - 1.0) > 1e-3:
            interp = cv2.INTER_LINEAR if resize > 1 else cv2.INTER_AREA
            img = cv2.resize(frame, (round(w0 * resize), round(h0 * resize)), interpolation=interp)
        else:
            img = frame
        rh, rw = img.shape[:2]

        blob = img.astype(np.float32)
        blob -= _BGR_MEAN
        blob = blob.transpose(2, 0, 1)[None]  # 1xCxHxW
        tensor = torch.from_numpy(np.ascontiguousarray(blob)).to(self._torch_device)

        loc, conf, landm = self.model(tensor)

        priors = PriorBox(self.cfg, (rh, rw)).forward().to(self._torch_device)
        variance = self.cfg["variance"]

        boxes = decode(loc.squeeze(0), priors, variance)
        box_scale = torch.tensor([rw, rh, rw, rh], device=self._torch_device, dtype=torch.float32)
        boxes = boxes * box_scale / resize

        landmarks = decode_landm(landm.squeeze(0), priors, variance)
        landm_scale = torch.tensor([rw, rh] * 5, device=self._torch_device, dtype=torch.float32)
        landmarks = landmarks * landm_scale / resize

        scores = conf.squeeze(0)[:, 1]

        keep_mask = scores > confidence
        boxes, landmarks, scores = boxes[keep_mask], landmarks[keep_mask], scores[keep_mask]
        if scores.numel() == 0:
            return []

        top_k = settings.retinaface_top_k
        if scores.numel() > top_k:
            order = torch.topk(scores, top_k).indices
            boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        keep = nms(boxes, scores, settings.nms_iou)[: settings.retinaface_keep_top_k]
        boxes = boxes[keep].cpu().numpy()
        landmarks = landmarks[keep].cpu().numpy().reshape(-1, 5, 2)
        scores = scores[keep].cpu().numpy()

        out: list[Detection] = []
        for (x1, y1, x2, y2), lm, s in zip(boxes, landmarks, scores, strict=True):
            det = Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                score=float(s),
                landmarks=lm.astype(np.float32),
            ).clip(w0, h0)
            out.append(det)
        return out
