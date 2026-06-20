#!/usr/bin/env python3
"""Run the two proposed methods across multiple seeds and report results."""

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
    parser.add_argument("--dataset", default="etis")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seeds", default="42,1,2")
    parser.add_argument("--output-root", default="outputs_apdr_ablation/etis")
    parser.add_argument("--allow-insecure-download", action="store_true")
    parser.add_argument("--delete-checkpoints-after-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-tests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    py = sys.executable
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    tables = output_root / "results" / "tables"
    summary = tables / "multi_seed_summary.csv"
    training = tables / "two_proposal_training_summary.csv"
    latex = tables / "two_proposal_comparison.tex"
    deltas = tables / "two_proposal_deltas.csv"

    run([py, str(PROJECT_ROOT / "tools" / "audit_repository_cleanliness.py")])
    run([py, str(PROJECT_ROOT / "tools" / "audit_fairness.py")])
    if args.run_tests:
        run([
            py,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_apdr_fourier_unet.py",
            "tests/test_repository_scope_and_fairness.py",
            "tests/test_pipeline_contracts.py",
        ])

    if not args.skip_training:
        benchmark = [
            py,
            str(PROJECT_ROOT / "scripts" / "benchmark_multi_seed.py"),
            "--models",
            MODELS,
            "--config-dir",
            "configs/ablation",
            "--dataset",
            args.dataset,
            "--data-root",
            args.data_root,
            "--image-size",
            str(args.image_size),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--device",
            args.device,
            "--num-workers",
            str(args.num_workers),
            "--seeds",
            args.seeds,
            "--output-root",
            str(output_root),
        ]
        if args.allow_insecure_download:
            benchmark.append("--allow-insecure-download")
        if args.delete_checkpoints_after_eval:
            benchmark.append("--delete-checkpoints-after-eval")
        run(benchmark)

    run([
        py,
        str(PROJECT_ROOT / "scripts" / "aggregate_training_results.py"),
        "--output-root",
        str(output_root),
        "--output-path",
        str(training),
        "--models",
        MODELS,
        "--seeds",
        args.seeds,
        "--dataset",
        args.dataset,
        "--expected-batch-size",
        str(args.batch_size),
    ])
    run([
        py,
        str(PROJECT_ROOT / "scripts" / "validate_ablation_results.py"),
        "--summary-path",
        str(summary),
        "--training-summary-path",
        str(training),
        "--output-root",
        str(output_root),
        "--models",
        MODELS,
        "--seeds",
        args.seeds,
        "--dataset",
        args.dataset,
    ])
    run([
        py,
        str(PROJECT_ROOT / "scripts" / "report_two_proposal_comparison.py"),
        "--summary-path",
        str(summary),
        "--latex-path",
        str(latex),
        "--delta-path",
        str(deltas),
    ])

    print("Two-proposal comparison completed successfully.")
    print(f"Test summary: {summary}")
    print(f"Training summary: {training}")
    print(f"LaTeX table: {latex}")
    print(f"Deltas: {deltas}")


if __name__ == "__main__":
    main()
