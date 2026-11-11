#!/usr/bin/env python
"""Print Torch / accelerator info (CUDA or ROCm-via-cuda)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from pc_mnist.utils import describe_device, resolve_device


def main() -> None:
    print(describe_device())
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"hip={getattr(torch.version, 'hip', None)}")
    device = resolve_device()
    if device.type != "cuda":
        print(
            "WARNING: GPU not visible. On ModelScope AMD instances, "
            "use the ROCm image torch and avoid `pip install torch`."
        )
        sys.exit(1)

    # Tiny device smoke: alloc + matmul on GPU
    x = torch.randn(1024, 1024, device=device)
    y = x @ x
    print(f"matmul_ok=True sum={float(y.sum()):.3f}")
    print("OK")


if __name__ == "__main__":
    main()
