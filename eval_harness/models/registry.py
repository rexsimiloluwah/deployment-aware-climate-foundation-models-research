from __future__ import annotations

import torch
from torch import nn


class ClassificationToySmokeModel(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], num_classes: int):
        super().__init__()
        in_features = 1
        for dim in input_shape:
            in_features *= dim
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(in_features, 128), nn.ReLU())
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class SegmentationToySmokeModel(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], num_classes: int):
        super().__init__()
        channels = input_shape[0]
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def build_smoke_model(dataset_cfg: dict) -> nn.Module:
    input_shape = tuple(dataset_cfg["input"]["shape"])
    num_classes = int(dataset_cfg["num_classes"])
    task_type = dataset_cfg["task_type"]
    if task_type == "classification":
        return ClassificationToySmokeModel(input_shape, num_classes)
    if task_type == "segmentation":
        return SegmentationToySmokeModel(input_shape, num_classes)
    raise ValueError(f"Unsupported task type: {task_type}")
