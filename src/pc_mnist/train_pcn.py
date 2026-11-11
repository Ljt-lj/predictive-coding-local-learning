from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .metrics import summarize_run
from .models_pcn import EnergyProbeCallback, build_pcn_model, count_pcn_parameters, make_train_config
from .utils import save_json


def train_pcn_experiment(
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    from pcn_torch import train_pcn, test_pcn

    model = build_pcn_model(
        hidden=list(cfg.get("hidden", [256, 128])),
        num_classes=int(cfg.get("num_classes", 10)),
        activation=str(cfg.get("activation", "relu")),
    ).to(device)

    probe = EnergyProbeCallback()
    epochs = int(cfg.get("epochs", 10))
    history_rows: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        callback = probe if epoch == 1 else None
        epoch_cfg_dict = {**cfg, "epochs": 1}
        train_config = make_train_config(epoch_cfg_dict, callback=callback)

        train_hist = train_pcn(model, train_loader, train_config)
        test_res = test_pcn(model, test_loader, train_config)

        train_acc = train_hist.train_accuracy[-1] if train_hist.train_accuracy else 0.0
        train_loss = train_hist.train_loss[-1] if train_hist.train_loss else 0.0
        test_acc = float(test_res["accuracy"])
        epoch_time = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "train_acc": float(train_acc),
            "test_acc": test_acc,
            "train_loss": float(train_loss),
            "epoch_time_sec": epoch_time,
        }
        history_rows.append(row)
        print(
            f"[PCN T={cfg.get('T_infer')}] epoch={epoch} "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
            f"energy={train_loss:.4f} time={epoch_time:.1f}s"
        )

    total_time = time.perf_counter() - total_start
    _write_history_csv(run_dir / "history.csv", history_rows)
    energy_steps = probe.steps
    _write_energy_csv(run_dir / "energy.csv", energy_steps)

    metrics = summarize_run(
        run_id=run_id,
        method="pcn",
        history=history_rows,
        cfg=cfg,
        device=str(device),
        param_count=count_pcn_parameters(model),
        total_train_time_sec=total_time,
        extra={
            "energy_probe_steps": len(energy_steps),
            "energy_probe_note": "first-batch inference energy of epoch 1",
            "final_test_energy": float(test_res.get("energy", 0.0)),
        },
    )
    save_json(metrics, run_dir / "metrics.json")
    torch.save(model.state_dict(), run_dir / "model.pt")
    return {
        "history": history_rows,
        "energy": energy_steps,
        "metrics": metrics,
        "model": model,
    }


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    fieldnames = ["epoch", "train_acc", "test_acc", "train_loss", "epoch_time_sec"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _write_energy_csv(path: Path, steps: list[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "energy"])
        writer.writeheader()
        for i, energy in enumerate(steps):
            writer.writerow({"step": i, "energy": energy})
