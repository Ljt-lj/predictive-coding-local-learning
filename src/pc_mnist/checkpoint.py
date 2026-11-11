from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from .utils import project_root, save_json


STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"


def resolve_runs_root(runs_dir: str | Path) -> Path:
    root = Path(runs_dir)
    if not root.is_absolute():
        root = project_root() / root
    return root


def status_path(run_dir: Path) -> Path:
    return run_dir / "status.json"


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint.pt"


def write_status(
    run_dir: Path,
    *,
    status: str,
    last_epoch: int,
    target_epochs: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "last_epoch": last_epoch,
        "target_epochs": target_epochs,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    save_json(payload, status_path(run_dir))


def read_status(run_dir: Path) -> dict[str, Any] | None:
    path = status_path(run_dir)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_run_complete(run_dir: Path) -> bool:
    st = read_status(run_dir)
    if st and st.get("status") == STATUS_COMPLETED and (run_dir / "metrics.json").exists():
        return True
    # Backward compatible: finished runs that only have metrics.json
    if (run_dir / "metrics.json").exists() and not checkpoint_path(run_dir).exists():
        return True
    return False


def is_run_resumable(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False
    if is_run_complete(run_dir):
        return False
    return checkpoint_path(run_dir).exists()


def save_checkpoint(run_dir: Path, payload: dict[str, Any]) -> None:
    torch.save(payload, checkpoint_path(run_dir))


def load_checkpoint(run_dir: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = checkpoint_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location=map_location)


def run_prefix(method: str, cfg: dict[str, Any]) -> str:
    custom = cfg.get("run_prefix")
    if custom:
        return f"{custom}_"
    if method == "bp":
        return "bp_"
    t = cfg.get("T_infer")
    if t is not None:
        return f"pcn_t{t}_"
    return "pcn_"


def list_runs_with_prefix(runs_dir: str | Path, prefix: str) -> list[Path]:
    root = resolve_runs_root(runs_dir)
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def find_latest_resumable_run(runs_dir: str | Path, method: str, cfg: dict[str, Any]) -> Path | None:
    for run_dir in list_runs_with_prefix(runs_dir, run_prefix(method, cfg)):
        if is_run_resumable(run_dir):
            return run_dir
    return None


def find_latest_complete_run(runs_dir: str | Path, method: str, cfg: dict[str, Any]) -> Path | None:
    for run_dir in list_runs_with_prefix(runs_dir, run_prefix(method, cfg)):
        if is_run_complete(run_dir):
            return run_dir
    return None


def load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
