from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import (
    STATUS_COMPLETED,
    STATUS_RUNNING,
    load_checkpoint,
    save_checkpoint,
    write_status,
)
from .logging_utils import setup_run_logger
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
    resume: bool = False,
) -> dict[str, Any]:
    logger = setup_run_logger(run_dir, name=f"bp:{run_id}")
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
    start_epoch = 1
    elapsed_before = 0.0
    best_test_acc = -1.0
    best_state = None

    if resume:
        ckpt = load_checkpoint(run_dir, map_location=device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        history = list(ckpt.get("history") or [])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        elapsed_before = float(ckpt.get("elapsed_sec", 0.0))
        best_test_acc = float(ckpt.get("best_test_acc", -1.0))
        if ckpt.get("best_state") is not None:
            best_state = ckpt["best_state"]
        logger.info(
            "Resuming BP from checkpoint: next_epoch=%s/%s history_len=%s",
            start_epoch,
            epochs,
            len(history),
        )
        if start_epoch > epochs:
            logger.info("Checkpoint already reached target epochs; finalizing.")
            return _finalize_bp(
                model=model,
                history=history,
                cfg=cfg,
                device=device,
                run_dir=run_dir,
                run_id=run_id,
                total_time=elapsed_before,
                best_state=best_state,
                logger=logger,
            )

    write_status(
        run_dir,
        status=STATUS_RUNNING,
        last_epoch=start_epoch - 1,
        target_epochs=epochs,
        extra={"method": "bp", "run_id": run_id},
    )
    logger.info("Start BP training epochs=%s device=%s resume=%s", epochs, device, resume)

    total_start = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
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
        logger.info(
            "[BP] epoch=%s train_acc=%.4f test_acc=%.4f loss=%.4f time=%.1fs",
            epoch,
            train_acc,
            test_acc,
            train_loss,
            epoch_time,
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, run_dir / "model_best.pt")

        elapsed = elapsed_before + (time.perf_counter() - total_start)
        save_checkpoint(
            run_dir,
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "best_test_acc": best_test_acc,
                "best_state": best_state,
                "elapsed_sec": elapsed,
            },
        )
        _write_history_csv(run_dir / "history.csv", history)
        write_status(
            run_dir,
            status=STATUS_RUNNING,
            last_epoch=epoch,
            target_epochs=epochs,
            extra={"method": "bp", "run_id": run_id, "best_test_acc": best_test_acc},
        )

    total_time = elapsed_before + (time.perf_counter() - total_start)
    return _finalize_bp(
        model=model,
        history=history,
        cfg=cfg,
        device=device,
        run_dir=run_dir,
        run_id=run_id,
        total_time=total_time,
        best_state=best_state,
        logger=logger,
    )


def _finalize_bp(
    *,
    model: nn.Module,
    history: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    run_id: str,
    total_time: float,
    best_state: dict[str, Any] | None,
    logger,
) -> dict[str, Any]:
    if best_state is not None:
        model.load_state_dict(best_state)

    _write_history_csv(run_dir / "history.csv", history)
    metrics = summarize_run(
        run_id=run_id,
        method="bp",
        history=history,
        cfg=cfg,
        device=str(device),
        param_count=count_parameters(model),
        total_train_time_sec=total_time,
        extra={"restored_best_checkpoint": best_state is not None},
    )
    save_json(metrics, run_dir / "metrics.json")
    torch.save(model.state_dict(), run_dir / "model.pt")
    write_status(
        run_dir,
        status=STATUS_COMPLETED,
        last_epoch=history[-1]["epoch"] if history else 0,
        target_epochs=int(cfg.get("epochs", 10)),
        extra={"method": "bp", "run_id": run_id, "best_test_acc": metrics["best_test_acc"]},
    )
    # Remove volatile checkpoint after successful completion (keeps model.pt / metrics)
    ckpt = run_dir / "checkpoint.pt"
    if ckpt.exists():
        ckpt.unlink()
    logger.info("BP done. best_test_acc=%.4f", metrics["best_test_acc"])
    return {"history": history, "energy": None, "metrics": metrics, "model": model}


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    fieldnames = ["epoch", "train_acc", "test_acc", "train_loss", "epoch_time_sec"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
