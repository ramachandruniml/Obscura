"""RetinaFace backbone configs (from biubug6/Pytorch_Retinaface data/config.py)."""

from __future__ import annotations

from typing import Any

cfg_mnet: dict[str, Any] = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "in_channel": 32,
    "out_channel": 64,
    "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
}

cfg_re50: dict[str, Any] = {
    "name": "Resnet50",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "in_channel": 256,
    "out_channel": 256,
    "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
}

BACKBONES = {"mobile0.25": cfg_mnet, "resnet50": cfg_re50}
