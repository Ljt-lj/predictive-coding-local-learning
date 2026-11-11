from __future__ import annotations

from typing import Any

import torch


def evaluate_accuracy(predict_fn, loader, device: torch.device) -> float:
    """Generic accuracy helper; predict_fn(batch_x) -> class indices."""
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = predict_fn(x)
        if pred.ndim > 1:
            pred = pred.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def history_to_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(history)
