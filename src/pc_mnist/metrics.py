from __future__ import annotations

from typing import Any


def epochs_to_threshold(history: list[dict[str, Any]], threshold: float = 0.90) -> int | None:
    for row in history:
        if float(row.get("test_acc", 0.0)) >= threshold:
            return int(row["epoch"])
    return None


def mean_epoch_time(history: list[dict[str, Any]]) -> float:
    times = [float(row["epoch_time_sec"]) for row in history if "epoch_time_sec" in row]
    if not times:
        return 0.0
    return sum(times) / len(times)


def summarize_run(
    *,
    run_id: str,
    method: str,
    history: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: str,
    param_count: int,
    total_train_time_sec: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    test_accs = [float(r["test_acc"]) for r in history] if history else [0.0]
    summary: dict[str, Any] = {
        "run_id": run_id,
        "method": method,
        "T_infer": cfg.get("T_infer"),
        "T_infer_test": cfg.get("T_infer_test"),
        "seed": cfg.get("seed"),
        "device": device,
        "param_count": param_count,
        "epochs": cfg.get("epochs"),
        "best_test_acc": max(test_accs),
        "final_test_acc": test_accs[-1],
        "epochs_to_90pct": epochs_to_threshold(history, 0.90),
        "mean_epoch_time_sec": mean_epoch_time(history),
        "total_train_time_sec": total_train_time_sec,
    }
    if extra:
        summary.update(extra)
    return summary
