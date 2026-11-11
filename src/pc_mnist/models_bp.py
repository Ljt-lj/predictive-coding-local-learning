from __future__ import annotations

import torch
import torch.nn as nn


def build_bp_mlp(
    input_dim: int = 784,
    hidden: list[int] | None = None,
    num_classes: int = 10,
) -> nn.Module:
    if hidden is None:
        hidden = [256, 128]
    layers: list[nn.Module] = []
    prev = input_dim
    for width in hidden:
        layers.append(nn.Linear(prev, width))
        layers.append(nn.ReLU())
        prev = width
    layers.append(nn.Linear(prev, num_classes))
    return nn.Sequential(*layers)


@torch.no_grad()
def bp_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    non_blocking = device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)
