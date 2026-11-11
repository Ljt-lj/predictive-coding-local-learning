from __future__ import annotations

from typing import Any

import torch
from torch import nn


def build_pcn_model(
    hidden: list[int] | None = None,
    num_classes: int = 10,
    activation: str = "tanh",
) -> nn.Module:
    """Build a PredictiveCodingNetwork aligned with BP MLP widths.

    Mapping: dims=[784, *hidden], output_dim=num_classes.
    Example: hidden=[256, 128] -> 784 -> 256 -> 128, readout to 10.

    Note: prefer ``tanh`` over ``relu`` for this generative PCN formulation;
    ReLU's zero derivative blocks local error flow and often stalls near chance.
    """
    try:
        from pcn_torch import PredictiveCodingNetwork
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pcn-torch is required for PCN experiments. "
            "Install with: pip install pcn-torch"
        ) from exc

    if hidden is None:
        hidden = [256, 128]
    dims = [784, *list(hidden)]
    model = PredictiveCodingNetwork(
        dims=dims,
        output_dim=num_classes,
        activation=activation,
        mode="classification",
    )
    return model


def resolve_lr_learn(cfg: dict[str, Any], epoch: int) -> float:
    """Resolve per-epoch lr_learn from schedule or simple decay knobs."""
    base = float(cfg.get("lr_learn", 0.005))
    schedule = cfg.get("lr_learn_schedule")
    if schedule:
        current = base
        for step in sorted(schedule, key=lambda s: int(s["epoch"])):
            if epoch >= int(step["epoch"]):
                current = float(step["lr"])
        return current

    decay_epoch = cfg.get("lr_decay_epoch")
    if decay_epoch is not None and epoch >= int(decay_epoch):
        return base * float(cfg.get("lr_decay_factor", 0.3))
    return base


def make_train_config(cfg: dict[str, Any], callback=None, *, epoch: int | None = None):
    from pcn_torch import TrainConfig

    t_infer = int(cfg.get("T_infer", 50))
    t_infer_test = cfg.get("T_infer_test")
    if t_infer_test is None:
        t_infer_test = max(200, 2 * t_infer)
    else:
        t_infer_test = int(t_infer_test)

    if epoch is None:
        lr_learn = float(cfg.get("lr_learn", 0.005))
    else:
        lr_learn = resolve_lr_learn(cfg, epoch)

    kwargs: dict[str, Any] = {
        "task": "classification",
        "T_infer": t_infer,
        "T_infer_test": t_infer_test,
        "lr_infer": float(cfg.get("lr_infer", 0.05)),
        "lr_learn": lr_learn,
        "num_epochs": int(cfg.get("epochs", 10)),
        "use_mixed_precision": bool(cfg.get("use_mixed_precision", True)),
        "callback": callback,
    }
    if cfg.get("T_learn") is not None:
        kwargs["T_learn"] = int(cfg["T_learn"])
    return TrainConfig(**kwargs)


def count_pcn_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EnergyProbeCallback:
    """Capture per-step inference energy for the first training batch."""

    def __init__(self) -> None:
        self.steps: list[float] = []
        self._capturing = False
        self._captured = False
        self._inner = None
        try:
            from pcn_torch import RichCallback

            self._inner = RichCallback()
        except Exception:
            self._inner = None

    def on_train_start(self, config, history) -> None:
        if self._inner is not None:
            self._inner.on_train_start(config, history)

    def on_epoch_start(self, epoch: int, num_epochs: int) -> None:
        if self._inner is not None:
            self._inner.on_epoch_start(epoch, num_epochs)

    def on_batch_start(self, batch: int, num_batches: int) -> None:
        if not self._captured:
            self._capturing = True
            self.steps = []
        if self._inner is not None:
            self._inner.on_batch_start(batch, num_batches)

    def on_inference_step(self, step: int, energy: float) -> None:
        if self._capturing:
            self.steps.append(float(energy))
        if self._inner is not None:
            self._inner.on_inference_step(step, energy)

    def on_learning_step(self, step: int, energy: float) -> None:
        if self._inner is not None:
            self._inner.on_learning_step(step, energy)

    def on_batch_end(self, batch: int, batch_energy: float, pre_learn_energy: float) -> None:
        if self._capturing:
            self._captured = True
            self._capturing = False
        if self._inner is not None:
            self._inner.on_batch_end(batch, batch_energy, pre_learn_energy)

    def on_epoch_end(self, epoch: int, history) -> None:
        if self._inner is not None:
            self._inner.on_epoch_end(epoch, history)

    def on_lr_reduced(self, old_lr: float, new_lr: float) -> None:
        if self._inner is not None:
            self._inner.on_lr_reduced(old_lr, new_lr)

    def on_train_end(self, history) -> None:
        if self._inner is not None:
            self._inner.on_train_end(history)
