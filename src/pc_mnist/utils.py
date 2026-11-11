from __future__ import annotations

import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def accelerator_backend() -> str:
    """Return a short label for the active Torch accelerator stack."""
    hip = getattr(torch.version, "hip", None)
    if hip:
        return f"rocm/hip ({hip})"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(requested: str | None = None) -> torch.device:
    """Pick a torch device.

    Note: PyTorch ROCm builds still expose AMD GPUs via ``torch.cuda`` /
    device type ``cuda`` (HIP compatibility layer). Prefer ``cuda`` on
    ModelScope AMD/ROCm images, not a separate ``hip`` device string.
    """
    if requested:
        name = requested.strip().lower()
        if name in {"gpu", "rocm", "hip"}:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Requested '{requested}' but torch.cuda.is_available() is False. "
                    "On ROCm images, the preinstalled torch must be the ROCm build; "
                    "do not overwrite it with a CPU/CUDA wheel."
                )
            return torch.device("cuda")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device | None = None) -> str:
    """Human-readable accelerator summary for logs / check scripts."""
    if device is None:
        device = resolve_device()
    parts = [
        f"device={device}",
        f"backend={accelerator_backend()}",
        f"torch={torch.__version__}",
    ]
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else 0
        parts.append(f"name={torch.cuda.get_device_name(idx)}")
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024**3)
        parts.append(f"vram={total_gb:.1f}GiB")
    return " | ".join(parts)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                cfg[key] = value
    return cfg


def make_run_id(method: str, extras: dict[str, Any] | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [method]
    if extras:
        if "T_infer" in extras and extras["T_infer"] is not None:
            parts.append(f"t{extras['T_infer']}")
    parts.append(stamp)
    return "_".join(str(p) for p in parts)


def prepare_run_dir(runs_dir: str | Path, run_id: str) -> Path:
    root = Path(runs_dir)
    if not root.is_absolute():
        root = project_root() / root
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    return run_dir


def save_config(cfg: dict[str, Any], run_dir: Path) -> None:
    path = run_dir / "config.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def copy_figures_to_repo(run_figures: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if not run_figures.exists():
        return
    for src in run_figures.glob("*.png"):
        shutil.copy2(src, dest / src.name)
