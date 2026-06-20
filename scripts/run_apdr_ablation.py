#!/usr/bin/env python3
"""Run the controlled ablation between the two proposed Fourier U-Net methods."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = "plain_fourier_unet,apdr_fourier_unet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="kvasir_seg")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="outputs_apdr_ablation")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-splits", action="store_true")
    parser.add_argument("--allow-insecure-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for audit in ("audit_repository_cleanliness.py", "audit_fairness.py"):
        audit_cmd = [sys.executable, str(PROJECT_ROOT / "tools" / audit)]
        print("[RUN]", " ".join(audit_cmd))
        subprocess.run(audit_cmd, cwd=PROJECT_ROOT, env=env, check=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "benchmark_all.py"),
        "--models", MODELS,
        "--config-dir", "configs/ablation",
        "--dataset", args.dataset,
        "--data-root", args.data_root,
        "--image-size", str(args.image_size),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--device", args.device,
        "--output-root", args.output_root,
        "--num-workers", str(args.num_workers),
        "--seed", str(args.seed),
    ]
    if args.skip_prepare:
        cmd.append("--skip-prepare")
    if args.skip_splits:
        cmd.append("--skip-splits")
    if args.allow_insecure_download:
        cmd.append("--allow-insecure-download")
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
