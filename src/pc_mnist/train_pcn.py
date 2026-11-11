from __future__ import annotations

import copy
import csv
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .checkpoint import (
    STATUS_COMPLETED,
    STATUS_RUNNING,
    load_checkpoint,
    save_checkpoint,
    write_status,
)
from .logging_utils import setup_run_logger
from .metrics import summarize_run
from .models_pcn import (
    EnergyProbeCallback,
    build_pcn_model,
    count_pcn_parameters,
    make_train_config,
    resolve_lr_learn,
)
from .utils import save_json


def train_pcn_experiment(
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    run_id: str,
    resume: bool = False,
) -> dict[str, Any]:
    from pcn_torch import train_pcn, test_pcn

    logger = setup_run_logger(run_dir, name=f"pcn:{run_id}")
    model = build_pcn_model(
        hidden=list(cfg.get("hidden", [256, 128])),
        num_classes=int(cfg.get("num_classes", 10)),
        activation=str(cfg.get("activation", "tanh")),
    ).to(device)

    probe = EnergyProbeCallback()
    epochs = int(cfg.get("epochs", 10))
    patience = cfg.get("early_stop_patience")
    patience = int(patience) if patience is not None else None
    history_rows: list[dict[str, Any]] = []
    energy_steps: list[float] = []
    start_epoch = 1
    elapsed_before = 0.0

    best_test_acc = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improve = 0
    test_res: dict[str, Any] = {"accuracy": 0.0, "energy": 0.0}

    if resume:
        ckpt = load_checkpoint(run_dir, map_location=device)
        model.load_state_dict(ckpt["model"])
        history_rows = list(ckpt.get("history") or [])
        energy_steps = list(ckpt.get("energy_steps") or [])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        elapsed_before = float(ckpt.get("elapsed_sec", 0.0))
        best_test_acc = float(ckpt.get("best_test_acc", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        best_state = ckpt.get("best_state")
        epochs_without_improve = int(ckpt.get("epochs_without_improve", 0))
        logger.info(
            "Resuming PCN from checkpoint: next_epoch=%s/%s T_infer=%s history_len=%s",
            start_epoch,
            epochs,
            cfg.get("T_infer"),
            len(history_rows),
        )
        if start_epoch > epochs:
            logger.info("Checkpoint already reached target epochs; finalizing.")
            return _finalize_pcn(
                model=model,
                history_rows=history_rows,
                energy_steps=energy_steps,
                cfg=cfg,
                device=device,
                run_dir=run_dir,
                run_id=run_id,
                total_time=elapsed_before,
                best_state=best_state,
                best_epoch=best_epoch,
                best_test_acc=best_test_acc,
                test_loader=test_loader,
                logger=logger,
            )

    write_status(
        run_dir,
        status=STATUS_RUNNING,
        last_epoch=start_epoch - 1,
        target_epochs=epochs,
        extra={"method": "pcn", "run_id": run_id, "T_infer": cfg.get("T_infer")},
    )
    logger.info(
        "Start PCN training epochs=%s T_infer=%s device=%s resume=%s",
        epochs,
        cfg.get("T_infer"),
        device,
        resume,
    )

    total_start = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        # Capture energy curve once: first epoch of a fresh run
        callback = probe if (epoch == 1 and not energy_steps) else None
        lr_learn = resolve_lr_learn(cfg, epoch)
        epoch_cfg_dict = {**cfg, "epochs": 1, "lr_learn": lr_learn}
        train_config = make_train_config(epoch_cfg_dict, callback=callback, epoch=epoch)
        logger.info(
            "[PCN T=%s] epoch=%s/%s lr_learn=%s T_learn=%s",
            cfg.get("T_infer"),
            epoch,
            epochs,
            lr_learn,
            cfg.get("T_learn"),
        )

        train_hist = train_pcn(model, train_loader, train_config)
        test_res = test_pcn(model, test_loader, train_config)

        train_acc = train_hist.train_accuracy[-1] if train_hist.train_accuracy else 0.0
        train_loss = train_hist.train_loss[-1] if train_hist.train_loss else 0.0
        test_acc = float(test_res["accuracy"])
        epoch_time = time.perf_counter() - epoch_start

        if callback is not None and probe.steps:
            energy_steps = list(probe.steps)
            _write_energy_csv(run_dir / "energy.csv", energy_steps)

        row = {
            "epoch": epoch,
            "train_acc": float(train_acc),
            "test_acc": test_acc,
            "train_loss": float(train_loss),
            "epoch_time_sec": epoch_time,
        }
        history_rows.append(row)
        logger.info(
            "[PCN T=%s] epoch=%s train_acc=%.4f test_acc=%.4f energy=%.4f time=%.1fs",
            cfg.get("T_infer"),
            epoch,
            train_acc,
            test_acc,
            train_loss,
            epoch_time,
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
            torch.save(best_state, run_dir / "model_best.pt")
        else:
            epochs_without_improve += 1

        elapsed = elapsed_before + (time.perf_counter() - total_start)
        save_checkpoint(
            run_dir,
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "history": history_rows,
                "energy_steps": energy_steps,
                "best_test_acc": best_test_acc,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "epochs_without_improve": epochs_without_improve,
                "elapsed_sec": elapsed,
            },
        )
        _write_history_csv(run_dir / "history.csv", history_rows)
        write_status(
            run_dir,
            status=STATUS_RUNNING,
            last_epoch=epoch,
            target_epochs=epochs,
            extra={
                "method": "pcn",
                "run_id": run_id,
                "T_infer": cfg.get("T_infer"),
                "best_test_acc": best_test_acc,
                "best_epoch": best_epoch,
            },
        )

        if patience is not None and epochs_without_improve >= patience:
            logger.info(
                "[PCN T=%s] early stop at epoch=%s (best_epoch=%s, best_test_acc=%.4f)",
                cfg.get("T_infer"),
                epoch,
                best_epoch,
                best_test_acc,
            )
            break

    total_time = elapsed_before + (time.perf_counter() - total_start)
    return _finalize_pcn(
        model=model,
        history_rows=history_rows,
        energy_steps=energy_steps,
        cfg=cfg,
        device=device,
        run_dir=run_dir,
        run_id=run_id,
        total_time=total_time,
        best_state=best_state,
        best_epoch=best_epoch,
        best_test_acc=best_test_acc,
        test_loader=test_loader,
        logger=logger,
    )


def _finalize_pcn(
    *,
    model,
    history_rows: list[dict[str, Any]],
    energy_steps: list[float],
    cfg: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    run_id: str,
    total_time: float,
    best_state,
    best_epoch: int,
    best_test_acc: float,
    test_loader: DataLoader,
    logger,
) -> dict[str, Any]:
    from pcn_torch import test_pcn

    test_res: dict[str, Any] = {"accuracy": best_test_acc, "energy": 0.0}
    if best_state is not None:
        model.load_state_dict(best_state)
        final_cfg = make_train_config({**cfg, "epochs": 1}, callback=None)
        test_res = test_pcn(model, test_loader, final_cfg)

    _write_history_csv(run_dir / "history.csv", history_rows)
    if energy_steps:
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
            "best_epoch": best_epoch,
            "restored_best_checkpoint": best_state is not None,
            "evaluated_test_acc_best_weights": float(test_res.get("accuracy", best_test_acc)),
        },
    )
    save_json(metrics, run_dir / "metrics.json")
    torch.save(model.state_dict(), run_dir / "model.pt")
    write_status(
        run_dir,
        status=STATUS_COMPLETED,
        last_epoch=history_rows[-1]["epoch"] if history_rows else 0,
        target_epochs=int(cfg.get("epochs", 10)),
        extra={
            "method": "pcn",
            "run_id": run_id,
            "T_infer": cfg.get("T_infer"),
            "best_test_acc": metrics["best_test_acc"],
            "best_epoch": best_epoch,
        },
    )
    ckpt = run_dir / "checkpoint.pt"
    if ckpt.exists():
        ckpt.unlink()
    logger.info(
        "PCN done. best_test_acc=%.4f best_epoch=%s",
        metrics["best_test_acc"],
        best_epoch,
    )
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
