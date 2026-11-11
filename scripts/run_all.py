#!/usr/bin/env python
"""Run BP + PCN (T=20/50/100) sequentially, with optional resume/skip."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full experiment suite")
    p.add_argument(
        "--cloud",
        action="store_true",
        help="Use configs/cloud_*.yaml (larger batch, more workers; for ROCm/GPU)",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--test-subset", type=int, default=None)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete runs; skip experiments that already completed",
    )
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    args = parse_args()
    py = sys.executable
    common: list[str] = []
    if args.epochs is not None:
        common += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        common += ["--batch-size", str(args.batch_size)]
    if args.device is not None:
        common += ["--device", args.device]
    if args.train_subset is not None:
        common += ["--train-subset", str(args.train_subset)]
    if args.test_subset is not None:
        common += ["--test-subset", str(args.test_subset)]
    if args.resume:
        common += ["--resume", "--skip-if-complete"]

    if args.cloud:
        bp_cfg = "configs/cloud_bp.yaml"
        pcn_cfgs = [
            "configs/cloud_pcn_t20.yaml",
            "configs/cloud_pcn_t50.yaml",
            "configs/cloud_pcn_t100.yaml",
        ]
    else:
        bp_cfg = "configs/bp.yaml"
        pcn_cfgs = [
            "configs/pcn_t20.yaml",
            "configs/pcn_t50.yaml",
            "configs/pcn_t100.yaml",
        ]

    run([py, "scripts/run_bp.py", "--config", bp_cfg, *common])
    for cfg in pcn_cfgs:
        run([py, "scripts/run_pcn.py", "--config", cfg, *common])
    run([py, "scripts/make_figures.py", "--runs-dir", "runs", "--out-dir", "figures"])
    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
