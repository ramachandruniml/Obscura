"""Prior-box generation and box/landmark decoding.

From biubug6/Pytorch_Retinaface (layers/functions/prior_box.py, utils/box_utils.py).
Only the inference-time functions are kept.
"""

from __future__ import annotations

from itertools import product
from math import ceil
from typing import Any

import torch


class PriorBox:
    """Generate anchor centers/sizes (normalized to [0, 1]) for a given input size."""

    def __init__(self, cfg: dict[str, Any], image_size: tuple[int, int]) -> None:
        self.min_sizes: list[list[int]] = cfg["min_sizes"]
        self.steps: list[int] = cfg["steps"]
        self.clip: bool = cfg["clip"]
        self.image_size = image_size  # (height, width)
        self.feature_maps = [
            [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
            for step in self.steps
        ]

    def forward(self) -> torch.Tensor:
        anchors: list[float] = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in (j + 0.5,)]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in (i + 0.5,)]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.tensor(anchors, dtype=torch.float32).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def decode(loc: torch.Tensor, priors: torch.Tensor, variances: list[float]) -> torch.Tensor:
    """Decode box regression offsets (from anchor form) into xyxy boxes (normalized)."""
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        dim=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre: torch.Tensor, priors: torch.Tensor, variances: list[float]) -> torch.Tensor:
    """Decode 5-point landmark regression offsets into normalized (x, y) pairs."""
    return torch.cat(
        (
            priors[:, :2] + pre[:, 0:2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        dim=1,
    )
