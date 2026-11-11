#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pc_mnist.plot import make_all_figures
from pc_mnist.utils import project_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate PPT figures from runs/")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default="figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    if not runs_dir.is_absolute():
        runs_dir = project_root() / runs_dir
    if not out_dir.is_absolute():
        out_dir = project_root() / out_dir

    paths = make_all_figures(runs_dir, out_dir)
    print("Wrote:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
