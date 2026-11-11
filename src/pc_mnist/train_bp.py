from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import summarize_run
from .models_bp import bp_accuracy, build_bp_mlp
from .utils import count_parameters, save_json


def train_bp(
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    model = build_bp_mlp(
        input_dim=784,
        hidden=list(cfg.get("hidden", [256, 128])),
        num_classes=int(cfg.get("num_classes", 10)),
    ).to(device)

    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    epochs = int(cfg.get("epochs", 10))
    history: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0
        correct = 0
        total = 0

        non_blocking = device.type == "cuda"
        for x, y in tqdm(train_loader, desc=f"BP epoch {epoch}/{epochs}", leave=False):
            x = x.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

        train_acc = correct / max(total, 1)
        train_loss = running_loss / max(total, 1)
        test_acc = bp_accuracy(model, test_loader, device)
        epoch_time = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "epoch_time_sec": epoch_time,
        }
        history.append(row)
        print(
            f"[BP] epoch={epoch} train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f} loss={train_loss:.4f} time={epoch_time:.1f}s"
        )

    total_time = time.perf_counter() - total_start
    _write_history_csv(run_dir / "history.csv", history)

    metrics = summarize_run(
        run_id=run_id,
        method="bp",
        history=history,
        cfg=cfg,
        device=str(device),
        param_count=count_parameters(model),
        total_train_time_sec=total_time,
    )
    save_json(metrics, run_dir / "metrics.json")
    torch.save(model.state_dict(), run_dir / "model.pt")
    return {"history": history, "energy": None, "metrics": metrics, "model": model}


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    fieldnames = ["epoch", "train_acc", "test_acc", "train_loss", "epoch_time_sec"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
