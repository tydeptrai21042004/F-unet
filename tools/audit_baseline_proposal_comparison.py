#!/usr/bin/env python3
"""Audit the fair 11-baseline versus 2-proposal comparison configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from baseline_proposal_spec import (  # noqa: E402
    BASELINE_MODELS,
    BASELINE_PROPOSAL_MODELS,
    PROPOSAL_MODELS,
)

SHARED_PATHS = [
    ("data", "augmentation"),
    ("data", "batch_size"),
    ("data", "image_size"),
    ("data", "num_workers"),
    ("data", "pin_memory"),
    ("train", "epochs"),
    ("train", "lr"),
    ("train", "weight_decay"),
    ("train", "optimizer"),
    ("train", "scheduler"),
    ("train", "t_max"),
    ("train", "eta_min"),
    ("train", "mixed_precision"),
    ("train", "deterministic"),
    ("train", "deterministic_warn_only"),
    ("train", "grad_clip"),
    ("train", "loss"),
    ("train", "threshold"),
    ("train", "aux_loss_weight"),
    ("train", "use_aux_outputs_loss"),
    ("train", "use_boundary_loss"),
    ("train", "gradient_accumulation_steps"),
    ("eval", "loss"),
    ("eval", "threshold"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="configs/fair")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return payload


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = PROJECT_ROOT / config_dir

    expected_files = {f"{name}.yaml" for name in BASELINE_PROPOSAL_MODELS}
    actual_files = {path.name for path in config_dir.glob("*.yaml")}
    config_file_mismatch = {
        "missing": sorted(expected_files - actual_files),
        "unexpected": sorted(actual_files - expected_files),
    }

    configs: dict[str, dict[str, Any]] = {}
    metadata_errors: list[str] = []
    for name in BASELINE_PROPOSAL_MODELS:
        path = config_dir / f"{name}.yaml"
        if not path.is_file():
            continue
        cfg = load(path)
        configs[name] = cfg
        if cfg.get("model", {}).get("name") != name:
            metadata_errors.append(
                f"{path.relative_to(PROJECT_ROOT)}: model.name="
                f"{cfg.get('model', {}).get('name')!r}, expected {name!r}"
            )
        if cfg.get("experiment", {}).get("name") != name:
            metadata_errors.append(
                f"{path.relative_to(PROJECT_ROOT)}: experiment.name="
                f"{cfg.get('experiment', {}).get('name')!r}, expected {name!r}"
            )

    strict_mismatches: dict[str, dict[str, Any]] = {}
    if "plain_fourier_unet" in configs:
        reference = configs["plain_fourier_unet"]
        for section, key in SHARED_PATHS:
            expected = reference.get(section, {}).get(key)
            values = {
                name: cfg.get(section, {}).get(key)
                for name, cfg in configs.items()
            }
            if any(value != expected for value in values.values()):
                strict_mismatches[f"{section}.{key}"] = values

    proposal_mismatches: dict[str, dict[str, Any]] = {}
    unexpected_apdr_keys: list[str] = []
    if all(name in configs for name in PROPOSAL_MODELS):
        plain = configs["plain_fourier_unet"]["model"]
        apdr = configs["apdr_fourier_unet"]["model"]
        proposal_mismatches = {
            key: {"plain": value, "apdr": apdr.get(key)}
            for key, value in plain.items()
            if key != "name" and apdr.get(key) != value
        }
        unexpected_apdr_keys = sorted(
            key
            for key in set(apdr) - set(plain)
            if not key.startswith("apdr_")
        )

    binary_segmentation_errors = []
    for name, cfg in configs.items():
        model_cfg = cfg.get("model", {})
        if model_cfg.get("in_channels") != 3:
            binary_segmentation_errors.append(f"{name}: in_channels must be 3")
        if model_cfg.get("num_classes") != 1:
            binary_segmentation_errors.append(f"{name}: num_classes must be 1")

    fair = not any(
        (
            config_file_mismatch["missing"],
            config_file_mismatch["unexpected"],
            metadata_errors,
            strict_mismatches,
            proposal_mismatches,
            unexpected_apdr_keys,
            binary_segmentation_errors,
        )
    )

    report = {
        "config_dir": str(config_dir),
        "baseline_models": BASELINE_MODELS,
        "proposal_models": PROPOSAL_MODELS,
        "all_models": BASELINE_PROPOSAL_MODELS,
        "config_file_mismatch": config_file_mismatch,
        "metadata_errors": metadata_errors,
        "strict_protocol_mismatches": strict_mismatches,
        "plain_vs_apdr_shared_path_mismatches": proposal_mismatches,
        "unexpected_apdr_keys": unexpected_apdr_keys,
        "binary_segmentation_errors": binary_segmentation_errors,
        "fair": fair,
        "notes": [
            "All 13 methods use one shared data, optimization, loss, epoch, threshold, and evaluation protocol.",
            "The fair comparison disables architecture-specific auxiliary and boundary losses for every method.",
            "APDR preserves every Plain Fourier model setting and adds only apdr_* adapter settings.",
        ],
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Fair baseline/proposal comparison: {fair}")
        print(f"Baselines: {len(BASELINE_MODELS)}")
        print(f"Proposals: {len(PROPOSAL_MODELS)}")
        if not fair:
            print(json.dumps(report, indent=2))

    if not fair:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
