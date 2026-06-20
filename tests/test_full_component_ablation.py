from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from scripts.component_ablation_spec import FULL_COMPONENT_ABLATION_MODELS
from src.models import build_model

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "full_component_ablation"


def load_config(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def build(name: str):
    config = load_config(name)
    model_config = dict(config["model"])
    model_config.pop("name", None)
    return build_model(name, config=model_config)


def main_logits(output):
    return output["main"] if isinstance(output, dict) else output


def test_all_permanent_configs_exist() -> None:
    assert {path.stem for path in CONFIG_DIR.glob("*.yaml")} == set(
        FULL_COMPONENT_ABLATION_MODELS
    )


def test_all_configs_share_identical_protocol() -> None:
    configs = [load_config(name) for name in FULL_COMPONENT_ABLATION_MODELS]
    reference = configs[0]
    for config in configs:
        assert config["data"] == reference["data"]
        assert config["train"] == reference["train"]
        assert config["eval"] == reference["eval"]


@pytest.mark.parametrize("name", FULL_COMPONENT_ABLATION_MODELS)
def test_every_component_ablation_model_builds(name: str) -> None:
    assert isinstance(build(name), torch.nn.Module)


@pytest.mark.parametrize(
    "name",
    [
        "unet",
        "plain_fourier_unet",
        "plain_fourier_amplitude_only",
        "plain_fourier_phase_only",
        "plain_fourier_no_channel_mix",
        "plain_fourier_no_residual",
        "apdr_uniform_route",
        "apdr_no_disagreement",
        "apdr_no_uncertainty",
        "apdr_no_boundary",
        "apdr_no_context",
        "apdr_local_amplitude_only",
        "apdr_local_phase_only",
        "apdr_fourier_unet",
    ],
)
def test_every_component_ablation_forward_backward(name: str) -> None:
    torch.manual_seed(7)
    model = build(name)
    model.train()
    if hasattr(model, "set_epoch"):
        model.set_epoch(5)
    x = torch.randn(1, 3, 64, 64)
    logits = main_logits(model(x))
    assert logits.shape == (1, 1, 64, 64)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()


def test_complete_apdr_equals_plain_at_zero_gate() -> None:
    torch.manual_seed(13)
    plain = build("plain_fourier_unet").eval()
    apdr = build("apdr_fourier_unet").eval()
    apdr.load_state_dict(plain.state_dict(), strict=False)
    assert apdr.apdr_adapter.beta_raw.item() == 0.0
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        plain_logits = main_logits(plain(x))
        apdr_logits = main_logits(apdr(x))
    torch.testing.assert_close(apdr_logits, plain_logits, rtol=0.0, atol=1.0e-6)


def test_apdr_beta_receives_first_step_gradient() -> None:
    torch.manual_seed(19)
    model = build("apdr_fourier_unet")
    model.train()
    model.set_epoch(5)
    logits = main_logits(model(torch.randn(1, 3, 64, 64)))
    logits.square().mean().backward()
    gradient = model.apdr_adapter.beta_raw.grad
    assert gradient is not None
    assert torch.isfinite(gradient)
    assert gradient.abs().item() > 0.0


def test_component_flags_are_active() -> None:
    assert build("apdr_uniform_route").uniform_route
    assert build("apdr_no_disagreement").ablate_disagreement
    assert build("apdr_no_uncertainty").ablate_uncertainty
    assert build("apdr_no_boundary").ablate_boundary
    assert build("apdr_no_context").ablate_context

    amplitude_only = build("apdr_local_amplitude_only")
    phase_only = build("apdr_local_phase_only")
    assert amplitude_only.apdr_adapter.local_adapter.phase_max == 0.0
    assert phase_only.apdr_adapter.local_adapter.amplitude_scale == 0.0


def test_odd_spatial_sizes_are_supported() -> None:
    model = build("apdr_fourier_unet").eval()
    with torch.no_grad():
        logits = main_logits(model(torch.randn(1, 3, 70, 74)))
    assert logits.shape == (1, 1, 70, 74)
    assert torch.isfinite(logits).all()
