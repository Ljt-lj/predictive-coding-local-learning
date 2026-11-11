#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pc_mnist.data import build_mnist_loaders
from pc_mnist.train_bp import train_bp
from pc_mnist.utils import (
    describe_device,
    load_config,
    make_run_id,
    prepare_run_dir,
    project_root,
    resolve_device,
    save_config,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BP MLP baseline on MNIST")
    p.add_argument("--config", default="configs/bp.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--test-subset", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_subset": args.train_subset,
        "test_subset": args.test_subset,
        "seed": args.seed,
    }
    cfg = load_config(args.config, overrides)
    cfg["method"] = "bp"

    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(args.device)

    data_dir = cfg.get("data_dir", "data")
    if not Path(data_dir).is_absolute():
        data_dir = project_root() / data_dir

    cfg.setdefault("input_scale", "unit")

    train_loader, test_loader = build_mnist_loaders(
        data_dir=data_dir,
        batch_size=int(cfg.get("batch_size", 128)),
        num_workers=int(cfg.get("num_workers", 0)),
        train_subset=cfg.get("train_subset"),
        test_subset=cfg.get("test_subset"),
        input_scale=str(cfg.get("input_scale", "unit")),
    )

    run_id = make_run_id("bp")
    run_dir = prepare_run_dir(cfg.get("runs_dir", "runs"), run_id)
    save_config(cfg, run_dir)

    print(describe_device(device))
    print(f"Run dir: {run_dir}")
    result = train_bp(train_loader, test_loader, cfg, device, run_dir, run_id)
    print(f"Done. best_test_acc={result['metrics']['best_test_acc']:.4f}")


if __name__ == "__main__":
    main()
