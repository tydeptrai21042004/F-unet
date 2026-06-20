from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.engine.output_utils import parse_model_output
from src.models import build_model
from src.models.proposal.apdr_fourier_unet import APDRFourierUNet
from src.models.proposal.fourier_unet import PlainFourierUNet

ROOT = Path(__file__).resolve().parents[1]


def small_plain_config() -> dict:
    return {
        "in_channels": 3,
        "num_classes": 1,
        "channels": [4, 8, 16, 32, 64],
        "fourier_alpha": 0.5,
        "fourier_alpha_start": 0.5,
        "fourier_alpha_warmup_epochs": 0,
        "fourier_expansion": 1.0,
        "fourier_dropout": 0.0,
        "fourier_block_norm": "gn",
        "fourier_block_act": "gelu",
        "fourier_init_hw": [2, 2],
        "fourier_amplitude_scale": 1.0,
        "fourier_phase_max": float(torch.pi),
        "fourier_use_amplitude": True,
        "fourier_use_phase": True,
        "fourier_use_channel_mixing": True,
        "fourier_residual": True,
        "fourier_zero_init_output": True,
        "norm": "bn",
        "act": "relu",
        "decoder_use_cbam": False,
    }


def small_apdr_config() -> dict:
    cfg = small_plain_config()
    cfg.update(
        {
            "apdr_window_size": 8,
            "apdr_expansion": 1.0,
            "apdr_amplitude_scale": 0.20,
            "apdr_phase_max": float(torch.pi / 4),
            "apdr_highpass_start": 0.20,
            "apdr_dropout": 0.0,
            "apdr_beta_max": 0.25,
            "apdr_routing_floor": 0.05,
            "apdr_detach_disagreement": False,
            "apdr_warmup_epochs": 2,
        }
    )
    return cfg


def test_only_plain_and_apdr_spectral_models_build() -> None:
    assert isinstance(build_model("plain_fourier_unet", small_plain_config()), PlainFourierUNet)
    assert isinstance(build_model("apdr_fourier_unet", small_apdr_config()), APDRFourierUNet)


def test_apdr_is_exact_plain_fourier_at_zero_initialized_adapter() -> None:
    torch.manual_seed(12)
    plain = build_model("plain_fourier_unet", small_plain_config())
    apdr = build_model("apdr_fourier_unet", small_apdr_config())
    missing, unexpected = apdr.load_state_dict(plain.state_dict(), strict=False)
    assert missing and all(key.startswith("apdr_adapter.") for key in missing)
    assert unexpected == []

    plain.eval()
    apdr.eval()
    apdr.set_epoch(2)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        plain_logits = plain(x)
        output = apdr(x)
    assert torch.equal(output["main"], output["baseline_logits"])
    assert torch.equal(output["main"], plain_logits)
    assert float(output["beta"].item()) == pytest.approx(0.0, abs=0.0)


def test_apdr_outputs_are_finite_and_parseable_for_non_window_multiple() -> None:
    model = build_model("apdr_fourier_unet", small_apdr_config()).eval()
    x = torch.randn(1, 3, 48, 40)
    with torch.no_grad():
        output = model(x)
    parsed = parse_model_output(output)
    assert parsed.main.shape == (1, 1, 48, 40)
    for key in ["main", "baseline_logits", "route", "disagreement", "uncertainty", "boundary_evidence"]:
        assert torch.isfinite(output[key]).all(), key
    assert output["route"].shape == (1, 1, 48, 40)
    assert output["disagreement"].shape == (1, 1, 48, 40)


def test_zero_gate_receives_gradient_on_first_step() -> None:
    torch.manual_seed(5)
    model = build_model("apdr_fourier_unet", small_apdr_config()).train()
    model.set_epoch(2)
    x = torch.randn(2, 3, 32, 32)
    target = torch.rand(2, 1, 32, 32)
    logits = model(x)["main"]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    grad = model.apdr_adapter.beta_raw.grad
    assert grad is not None
    assert torch.isfinite(grad)
    assert grad.abs().item() > 0.0


def test_nonzero_gate_activates_only_the_new_residual_path() -> None:
    torch.manual_seed(9)
    model = build_model("apdr_fourier_unet", small_apdr_config()).eval()
    model.set_epoch(2)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        zero_output = model(x)
        model.apdr_adapter.beta_raw.fill_(0.75)
        active_output = model(x)
    assert torch.equal(zero_output["main"], zero_output["baseline_logits"])
    assert not torch.equal(active_output["main"], active_output["baseline_logits"])
    # The stored baseline path is independent of the residual gate.
    assert torch.equal(zero_output["baseline_logits"], active_output["baseline_logits"])


def test_disagreement_is_derived_from_distinct_amplitude_and_phase_evidence() -> None:
    model = build_model("apdr_fourier_unet", small_apdr_config()).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        model.fourier_bottleneck.amplitude_logits.fill_(0.4)
        model.fourier_bottleneck.phase_logits.fill_(-0.3)
        output = model(x)
    disagreement = output["disagreement"]
    assert disagreement.max().item() > 0.0
    assert 0.0 <= disagreement.min().item() <= disagreement.max().item() <= 1.0


def test_ablation_configs_build_with_identical_shared_protocol() -> None:
    config_dir = ROOT / "configs" / "ablation"
    for name in ("plain_fourier_unet", "apdr_fourier_unet"):
        cfg = yaml.safe_load((config_dir / f"{name}.yaml").read_text(encoding="utf-8"))
        model_cfg = dict(cfg["model"])
        model_cfg["channels"] = [4, 8, 16, 32, 64]
        model_cfg["fourier_init_hw"] = [2, 2]
        if name == "apdr_fourier_unet":
            model_cfg["apdr_window_size"] = 8
        model = build_model(name, model_cfg).eval()
        with torch.no_grad():
            parsed = parse_model_output(model(torch.randn(1, 3, 32, 32)))
        assert parsed.main.shape == (1, 1, 32, 32)
