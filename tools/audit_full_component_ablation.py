#!/usr/bin/env python3
"""Audit permanent full-component ablation configs and model registration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.component_ablation_spec import FULL_COMPONENT_ABLATION_MODELS
from src.models.registry import MODEL_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        default="configs/full_component_ablation",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"ERROR: invalid YAML mapping: {path}")
    return payload


def changed_keys(reference: dict, current: dict) -> set[str]:
    return {
        key
        for key in set(reference) | set(current)
        if reference.get(key) != current.get(key)
    }


def main() -> None:
    args = parse_args()
    config_dir = PROJECT_ROOT / args.config_dir
    expected = set(FULL_COMPONENT_ABLATION_MODELS)
    found = {path.stem for path in config_dir.glob("*.yaml")}
    if found != expected:
        raise SystemExit(
            "ERROR: full component configuration scope mismatch.\n"
            f"Expected: {sorted(expected)}\nFound:    {sorted(found)}"
        )

    missing_registry = expected - set(MODEL_REGISTRY)
    if missing_registry:
        raise SystemExit(
            "ERROR: unregistered ablation models: "
            + ", ".join(sorted(missing_registry))
        )

    configs = {
        name: load(config_dir / f"{name}.yaml")
        for name in FULL_COMPONENT_ABLATION_MODELS
    }
    for name, config in configs.items():
        if config.get("model", {}).get("name") != name:
            raise SystemExit(f"ERROR: {name}.yaml has an incorrect model.name")

    reference = configs["plain_fourier_unet"]
    for name, config in configs.items():
        for section in ("data", "train", "eval"):
            if config.get(section) != reference.get(section):
                raise SystemExit(
                    f"ERROR: unfair protocol: {name} differs in {section}"
                )

    expected_protocol = {
        ("data", "image_size"): 352,
        ("data", "batch_size"): 6,
        ("data", "augmentation"): "strong",
        ("train", "epochs"): 30,
        ("train", "lr"): 0.0003,
        ("train", "optimizer"): "adamw",
        ("train", "scheduler"): "cosine",
        ("train", "loss"): "bce_dice",
        ("train", "threshold"): 0.5,
        ("train", "use_aux_outputs_loss"): False,
        ("train", "use_boundary_loss"): False,
        ("train", "gradient_accumulation_steps"): 1,
        ("eval", "loss"): "bce_dice",
        ("eval", "threshold"): 0.5,
    }
    for (section, key), expected_value in expected_protocol.items():
        actual = reference.get(section, {}).get(key)
        if actual != expected_value:
            raise SystemExit(
                f"ERROR: {section}.{key}={actual!r}; "
                f"expected {expected_value!r}"
            )

    plain_reference = reference["model"]
    plain_allowed = {
        "plain_fourier_amplitude_only": {"name", "fourier_use_phase"},
        "plain_fourier_phase_only": {"name", "fourier_use_amplitude"},
        "plain_fourier_no_channel_mix": {"name", "fourier_use_channel_mixing"},
        "plain_fourier_no_residual": {
            "name",
            "fourier_residual",
            "fourier_zero_init_output",
        },
        "plain_fourier_unet": {"name"},
    }
    for name, allowed in plain_allowed.items():
        unexpected = changed_keys(plain_reference, configs[name]["model"]) - allowed
        if unexpected:
            raise SystemExit(
                f"ERROR: {name} changes unexpected Proposal-I keys: "
                + ", ".join(sorted(unexpected))
            )

    apdr_reference = configs["apdr_fourier_unet"]["model"]
    apdr_allowed = {
        "apdr_uniform_route": {"name"},
        "apdr_no_disagreement": {"name"},
        "apdr_no_uncertainty": {"name"},
        "apdr_no_boundary": {"name"},
        "apdr_no_context": {"name"},
        "apdr_local_amplitude_only": {"name", "apdr_phase_max"},
        "apdr_local_phase_only": {"name", "apdr_amplitude_scale"},
        "apdr_fourier_unet": {"name"},
    }
    for name, allowed in apdr_allowed.items():
        unexpected = changed_keys(apdr_reference, configs[name]["model"]) - allowed
        if unexpected:
            raise SystemExit(
                f"ERROR: {name} changes unexpected Proposal-II keys: "
                + ", ".join(sorted(unexpected))
            )

    plain_keys = set(plain_reference) - {"name"}
    for name in FULL_COMPONENT_ABLATION_MODELS:
        if not name.startswith("apdr_"):
            continue
        current = configs[name]["model"]
        for key in plain_keys:
            if current.get(key) != plain_reference.get(key):
                raise SystemExit(
                    f"ERROR: {name} changes preserved Plain-Fourier key {key}"
                )

    print("Full component ablation audit passed.")
    print(f"Configurations: {len(configs)}")
    print("All data/train/eval sections are identical.")
    print("Each architecture variant changes only its declared component.")


if __name__ == "__main__":
    main()
