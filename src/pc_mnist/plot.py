from __future__ import annotations

import csv
import json
from pathlib import Path


def _load_metrics(run_dir: Path) -> dict:
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_history(run_dir: Path) -> list[dict]:
    with (run_dir / "history.csv").open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_energy(run_dir: Path) -> list[dict]:
    path = run_dir / "energy.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def discover_runs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    return sorted(
        [p for p in runs_dir.iterdir() if p.is_dir() and (p / "metrics.json").exists()],
        key=lambda p: p.name,
    )


def make_all_figures(runs_dir: Path, out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(runs_dir)
    if not runs:
        raise FileNotFoundError(f"No completed runs found under {runs_dir}")

    metrics_list = [(p, _load_metrics(p)) for p in runs]
    written: list[Path] = []

    fig1 = out_dir / "fig1_acc_curves.png"
    plt.figure(figsize=(8, 5))
    for run_dir, meta in metrics_list:
        hist = _load_history(run_dir)
        epochs = [int(r["epoch"]) for r in hist]
        accs = [float(r["test_acc"]) for r in hist]
        label = meta["method"].upper()
        if meta.get("T_infer") is not None:
            label = f"PCN T={meta['T_infer']}"
        alpha = 1.0
        if meta["method"] == "pcn" and meta.get("T_infer") not in (50, None):
            alpha = 0.45
        plt.plot(epochs, accs, marker="o", label=label, alpha=alpha)
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title("Test accuracy vs epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1, dpi=200)
    plt.close()
    written.append(fig1)

    fig2 = out_dir / "fig2_t_infer_bar.png"
    pcn_runs = [(d, m) for d, m in metrics_list if m["method"] == "pcn"]
    bp_runs = [(d, m) for d, m in metrics_list if m["method"] == "bp"]
    plt.figure(figsize=(8, 5))
    if pcn_runs:
        by_t: dict[int, dict] = {}
        for _, m in pcn_runs:
            t = int(m.get("T_infer") or 0)
            by_t[t] = m
        ts = sorted(by_t)
        accs = [by_t[t]["best_test_acc"] for t in ts]
        plt.bar([str(t) for t in ts], accs, color="#4C78A8", label="PCN best test acc")
    if bp_runs:
        bp_acc = max(m["best_test_acc"] for _, m in bp_runs)
        plt.axhline(bp_acc, color="#F58518", linestyle="--", label=f"BP best={bp_acc:.3f}")
    plt.xlabel("T_infer")
    plt.ylabel("Best test accuracy")
    plt.title("Effect of inference steps on PCN accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2, dpi=200)
    plt.close()
    written.append(fig2)

    fig3 = out_dir / "fig3_energy_curve.png"
    energy_run = None
    for run_dir, meta in metrics_list:
        if meta["method"] == "pcn" and meta.get("T_infer") == 50 and (run_dir / "energy.csv").exists():
            energy_run = run_dir
            break
    if energy_run is None:
        for run_dir, meta in metrics_list:
            if meta["method"] == "pcn" and (run_dir / "energy.csv").exists():
                energy_run = run_dir
                break
    plt.figure(figsize=(8, 5))
    if energy_run is not None:
        edf = _load_energy(energy_run)
        steps = [int(r["step"]) for r in edf]
        energies = [float(r["energy"]) for r in edf]
        plt.plot(steps, energies, color="#54A24B")
        plt.xlabel("Inference step")
        plt.ylabel("Energy")
        plt.title("PCN inference energy settling curve")
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "No energy.csv found", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig3, dpi=200)
    plt.close()
    written.append(fig3)

    fig4 = out_dir / "fig4_time_bar.png"
    selected: dict[str, dict] = {}
    for _, m in metrics_list:
        key = "BP" if m["method"] == "bp" else f"PCN T={m.get('T_infer')}"
        selected[key] = m

    def sort_key(k: str) -> tuple[int, int]:
        if k == "BP":
            return (0, 0)
        return (1, int(str(k).split("=")[1]))

    labels = sorted(selected, key=sort_key)
    times = [selected[k]["mean_epoch_time_sec"] for k in labels]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, times, color="#E45756")
    plt.ylabel("Mean epoch time (sec)")
    plt.title("Training time comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(fig4, dpi=200)
    plt.close()
    written.append(fig4)

    return written
