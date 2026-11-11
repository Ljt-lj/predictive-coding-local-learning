#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pc_mnist.checkpoint import (
    find_latest_complete_run,
    find_latest_resumable_run,
    is_run_complete,
    load_run_config,
)
from pc_mnist.data import build_mnist_loaders
from pc_mnist.train_pcn import train_pcn_experiment
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
    p = argparse.ArgumentParser(description="Train PCN on MNIST")
    p.add_argument("--config", default="configs/pcn_t50.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--T-infer", type=int, default=None)
    p.add_argument("--T-infer-test", type=int, default=None)
    p.add_argument("--T-learn", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--test-subset", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume latest incomplete PCN run for this T_infer (or --run-dir)",
    )
    p.add_argument(
        "--run-dir",
        default=None,
        help="Explicit run directory to resume",
    )
    p.add_argument(
        "--skip-if-complete",
        action="store_true",
        help="If a completed PCN run for this T_infer exists, skip",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "T_infer": args.T_infer,
        "T_infer_test": args.T_infer_test,
        "T_learn": args.T_learn,
        "train_subset": args.train_subset,
        "test_subset": args.test_subset,
        "seed": args.seed,
    }
    cfg = load_config(args.config, overrides)
    cfg["method"] = "pcn"
    cfg.setdefault("activation", "tanh")
    cfg.setdefault("input_scale", "bipolar")
    cfg.setdefault("lr_learn", 0.005)

    runs_dir = cfg.get("runs_dir", "runs")
    resume = False
    run_dir: Path | None = None

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = project_root() / run_dir
        resume = True
        saved = load_run_config(run_dir)
        if saved:
            for key, value in saved.items():
                if overrides.get(key) is None:
                    cfg[key] = value
        for key, value in overrides.items():
            if value is not None:
                cfg[key] = value
        run_id = run_dir.name
    elif args.resume:
        found = find_latest_resumable_run(runs_dir, "pcn", cfg)
        if found is not None:
            run_dir = found
            resume = True
            run_id = run_dir.name
            saved = load_run_config(run_dir)
            if saved:
                for key, value in saved.items():
                    cfg.setdefault(key, value)
            for key, value in overrides.items():
                if value is not None:
                    cfg[key] = value
            print(f"Resuming run: {run_dir}")
        elif args.skip_if_complete:
            done = find_latest_complete_run(runs_dir, "pcn", cfg)
            if done is not None:
                print(f"Skip PCN T={cfg.get('T_infer')}; already complete: {done}")
                return
            print("No resumable/complete PCN run found; starting a new run.")
        else:
            print("No resumable PCN run found; starting a new run.")
    elif args.skip_if_complete:
        done = find_latest_complete_run(runs_dir, "pcn", cfg)
        if done is not None:
            print(f"Skip PCN T={cfg.get('T_infer')}; already complete: {done}")
            return

    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(args.device)

    data_dir = cfg.get("data_dir", "data")
    if not Path(data_dir).is_absolute():
        data_dir = project_root() / data_dir

    train_loader, test_loader = build_mnist_loaders(
        data_dir=data_dir,
        batch_size=int(cfg.get("batch_size", 128)),
        num_workers=int(cfg.get("num_workers", 0)),
        train_subset=cfg.get("train_subset"),
        test_subset=cfg.get("test_subset"),
        input_scale=str(cfg.get("input_scale", "bipolar")),
    )

    if run_dir is None:
        run_id = make_run_id("pcn", {"T_infer": cfg.get("T_infer")})
        run_dir = prepare_run_dir(runs_dir, run_id)
        save_config(cfg, run_dir)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, run_dir)

    if resume and is_run_complete(run_dir):
        print(f"Run already complete: {run_dir}")
        return

    print(describe_device(device))
    print(f"T_infer={cfg.get('T_infer')} T_infer_test={cfg.get('T_infer_test')}")
    print(f"Run dir: {run_dir} (resume={resume})")
    result = train_pcn_experiment(
        train_loader, test_loader, cfg, device, run_dir, run_id, resume=resume
    )
    print(f"Done. best_test_acc={result['metrics']['best_test_acc']:.4f}")


if __name__ == "__main__":
    main()
