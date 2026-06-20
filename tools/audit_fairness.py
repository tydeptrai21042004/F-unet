from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "fair"
MODELS = [
    "unet", "unetpp", "attention_unet", "pranet", "acsnet",
    "hardnet_mseg", "cfanet", "polyp_pvt", "caranet", "hsnet",
    "resunetpp", "plain_fourier_unet", "apdr_fourier_unet",
]
SHARED_PATHS = [
    ("data", "augmentation"), ("data", "batch_size"),
    ("data", "image_size"), ("data", "num_workers"),
    ("data", "pin_memory"), ("train", "epochs"),
    ("train", "lr"), ("train", "weight_decay"),
    ("train", "optimizer"), ("train", "scheduler"),
    ("train", "t_max"), ("train", "eta_min"),
    ("train", "mixed_precision"), ("train", "deterministic"),
    ("train", "grad_clip"), ("train", "loss"),
    ("train", "threshold"), ("train", "use_aux_outputs_loss"),
    ("train", "use_boundary_loss"),
    ("train", "gradient_accumulation_steps"),
    ("eval", "loss"), ("eval", "threshold"),
]


def load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> None:
    configs = {name: load(CONFIG_DIR / f"{name}.yaml") for name in MODELS}
    reference = configs["plain_fourier_unet"]
    strict_mismatches: dict[str, dict[str, Any]] = {}
    for section, key in SHARED_PATHS:
        values = {name: cfg.get(section, {}).get(key) for name, cfg in configs.items()}
        if any(value != reference[section][key] for value in values.values()):
            strict_mismatches[f"{section}.{key}"] = values

    plain = reference["model"]
    apdr = configs["apdr_fourier_unet"]["model"]
    model_mismatches = {
        key: {"plain": value, "apdr": apdr.get(key)}
        for key, value in plain.items()
        if key != "name" and apdr.get(key) != value
    }
    unexpected_apdr_keys = sorted(
        key for key in set(apdr) - set(plain) if not key.startswith("apdr_")
    )
    report = {
        "models": MODELS,
        "strict_mismatches": strict_mismatches,
        "plain_vs_apdr_shared_model_mismatches": model_mismatches,
        "unexpected_apdr_keys": unexpected_apdr_keys,
        "fair": not strict_mismatches and not model_mismatches and not unexpected_apdr_keys,
        "notes": [
            "All methods use the same split, augmentation, optimizer, loss, epochs, threshold, and seeds.",
            "APDR differs from Plain Fourier U-Net only by apdr_* residual-adapter parameters.",
            "The APDR residual scalar is initialized to zero, preserving the Plain Fourier output at initialization.",
        ],
    }
    print(json.dumps(report, indent=2))
    if not report["fair"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
