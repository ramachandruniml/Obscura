"""RetinaFace detection head + backbone assembly.

From biubug6/Pytorch_Retinaface models/retinaface.py, inference paths only.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import torchvision.models._utils as _utils
from torch import nn
from torchvision import models

from app.detectors._retinaface.net import FPN, SSH, MobileNetV1


class ClassHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 2) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 2) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 2) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 10, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        if cfg["name"] == "mobilenet0.25":
            backbone: nn.Module = MobileNetV1()
        elif cfg["name"] == "Resnet50":
            backbone = models.resnet50(weights=None)
        else:  # pragma: no cover - guarded by config
            raise ValueError(f"unknown RetinaFace backbone: {cfg['name']}")

        self.body = _utils.IntermediateLayerGetter(backbone, cfg["return_layers"])
        in_channels = cfg["in_channel"]
        in_channels_list = [in_channels * 2, in_channels * 4, in_channels * 8]
        out_channels = cfg["out_channel"]

        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = nn.ModuleList([ClassHead(out_channels, 2) for _ in range(3)])
        self.BboxHead = nn.ModuleList([BboxHead(out_channels, 2) for _ in range(3)])
        self.LandmarkHead = nn.ModuleList([LandmarkHead(out_channels, 2) for _ in range(3)])

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.body(inputs)
        fpn = self.fpn(out)
        features = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]

        bbox = torch.cat([self.BboxHead[i](f) for i, f in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](f) for i, f in enumerate(features)], dim=1)
        landm = torch.cat([self.LandmarkHead[i](f) for i, f in enumerate(features)], dim=1)

        # inference: softmax over the 2-class (bg/face) logits
        return bbox, F.softmax(classifications, dim=-1), landm


def remove_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def load_retinaface(cfg: dict[str, Any], weights_path: str, device: torch.device) -> RetinaFace:
    """Build the model and load a biubug6-format checkpoint (handles the ``module.`` prefix)."""
    model = RetinaFace(cfg)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    state = remove_prefix(state, "module.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The MobileNetV1 classifier (avg/fc) is dropped by IntermediateLayerGetter,
    # so checkpoint keys for it are benign "unexpected" entries.
    unexpected = [k for k in unexpected if not k.startswith(("body.fc", "fc", "body.avg", "avg"))]
    if missing or unexpected:
        raise RuntimeError(
            f"RetinaFace checkpoint mismatch for {weights_path}: "
            f"missing={missing[:6]} unexpected={unexpected[:6]}"
        )
    model.eval().to(device)
    return model
