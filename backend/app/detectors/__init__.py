"""Face detection backends.

Public surface (import from here, not the submodules):

    Detection, Detector, InvalidFrameError, iou   -- the common interface
    get_detector_class, build_detector, get_cached_detector  -- selection

Concrete implementations (loaded lazily by the registry):
    retinaface.py    - biubug6 Pytorch_Retinaface (accuracy, 5-pt landmarks)
    yolov8_face.py   - Ultralytics YOLOv8-face (speed)
    onnx_detector.py - ONNX Runtime (DETECTOR_RUNTIME=onnx); consumes a raw
                       YOLOv8-face ONNX export, decode/NMS in numpy (_yolo_onnx.py)
"""

from app.detectors.base import Detection, Detector, InvalidFrameError, iou
from app.detectors.registry import (
    VALID_BACKENDS,
    build_detector,
    clear_cache,
    get_cached_detector,
    get_detector_class,
)

__all__ = [
    "VALID_BACKENDS",
    "Detection",
    "Detector",
    "InvalidFrameError",
    "build_detector",
    "clear_cache",
    "get_cached_detector",
    "get_detector_class",
    "iou",
]
