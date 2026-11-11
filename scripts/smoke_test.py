#!/usr/bin/env python
"""Fast end-to-end smoke test on MNIST subsets (CPU-friendly)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pc_mnist.data import build_mnist_loaders
from pc_mnist.train_bp import train_bp
from pc_mnist.train_pcn import train_pcn_experiment
from pc_mnist.utils import (
    describe_device,
    make_run_id,
    prepare_run_dir,
    project_root,
    resolve_device,
    save_config,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test BP + PCN on MNIST subsets")
    p.add_argument("--device", default=None)
    p.add_argument("--skip-pcn", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    device = resolve_device(args.device)
    data_dir = project_root() / "data"

    print(f"[smoke] {describe_device(device)}")

    bp_train, bp_test = build_mnist_loaders(
        data_dir=data_dir,
        batch_size=64,
        num_workers=0,
        train_subset=2000,
        test_subset=500,
        input_scale="unit",
    )

    bp_cfg = {
        "method": "bp",
        "seed": 42,
        "epochs": 2,
        "hidden": [256, 128],
        "num_classes": 10,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 64,
        "input_scale": "unit",
    }
    bp_run_id = make_run_id("smoke_bp")
    bp_dir = prepare_run_dir("runs", bp_run_id)
    save_config(bp_cfg, bp_dir)
    bp_result = train_bp(bp_train, bp_test, bp_cfg, device, bp_dir, bp_run_id)
    print(f"[smoke] BP best_test_acc={bp_result['metrics']['best_test_acc']:.4f}")

    if not args.skip_pcn:
        pcn_train, pcn_test = build_mnist_loaders(
            data_dir=data_dir,
            batch_size=64,
            num_workers=0,
            train_subset=2000,
            test_subset=500,
            input_scale="bipolar",
        )
        pcn_cfg = {
            "method": "pcn",
            "seed": 42,
            "epochs": 2,
            "hidden": [256, 128],
            "num_classes": 10,
            "T_infer": 20,
            "T_infer_test": 80,
            "T_learn": 16,
            "lr_infer": 0.05,
            "lr_learn": 0.005,
            "activation": "tanh",
            "input_scale": "bipolar",
            "batch_size": 64,
            "use_mixed_precision": device.type == "cuda",
        }
        pcn_run_id = make_run_id("smoke_pcn", {"T_infer": 20})
        pcn_dir = prepare_run_dir("runs", pcn_run_id)
        save_config(pcn_cfg, pcn_dir)
        pcn_result = train_pcn_experiment(
            pcn_train, pcn_test, pcn_cfg, device, pcn_dir, pcn_run_id
        )
        print(f"[smoke] PCN best_test_acc={pcn_result['metrics']['best_test_acc']:.4f}")
        print(f"[smoke] energy steps captured={len(pcn_result['energy'] or [])}")

    print("[smoke] OK")


if __name__ == "__main__":
    main()
