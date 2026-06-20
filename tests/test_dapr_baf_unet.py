from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from scripts.dapr_baf_ablation_spec import DAPR_BAF_ABLATION_MODELS
from src.models import build_model

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "dapr_baf_ablation"


def load_config(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def build(name: str):
    config = load_config(name)
    model_config = dict(config["model"])
    model_config["channels"] = [4, 8, 16, 32, 64]
    model_config["fourier_init_hw"] = [2, 2]
    model_config["baf_window_size"] = 8
    model_config["baf_stride"] = 8 if name == "dapr_baf_nonoverlap" else 4
    model_config.pop("name", None)
    return build_model(name, config=model_config)


def main_logits(output):
    return output["main"] if isinstance(output, dict) else output


def test_all_dapr_baf_ablation_configs_exist() -> None:
    assert {path.stem for path in CONFIG_DIR.glob("*.yaml")} == set(
        DAPR_BAF_ABLATION_MODELS
    )


def test_all_dapr_baf_configs_share_the_same_protocol() -> None:
    configs = [load_config(name) for name in DAPR_BAF_ABLATION_MODELS]
    reference = configs[-1]
    for config in configs:
        assert config["data"] == reference["data"]
        assert config["train"] == reference["train"]
        assert config["eval"] == reference["eval"]


@pytest.mark.parametrize("name", DAPR_BAF_ABLATION_MODELS)
def test_every_dapr_baf_variant_builds_and_runs(name: str) -> None:
    torch.manual_seed(7)
    model = build(name)
    model.train()
    if hasattr(model, "set_epoch"):
        model.set_epoch(1)
    x = torch.randn(1, 3, 32, 32)
    logits = main_logits(model(x))
    assert logits.shape == (1, 1, 32, 32)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()


def test_direct_reconstruction_has_no_spatial_bypass() -> None:
    model = build("dapr_baf_unet")
    assert model.fourier_bottleneck.residual is False
    assert model.fourier_bottleneck.zero_init_output is False


def test_complete_model_uses_boundary_guided_overlapping_amplitude_refinement() -> None:
    model = build("dapr_baf_unet")
    refiner = model.baf_refiner
    assert refiner is not None
    assert refiner.use_boundary is True
    assert refiner.local_adapter.stride < refiner.local_adapter.window_size
    assert not hasattr(refiner.local_adapter, "phase_logits")
    assert refiner.effective_beta().detach().item() > 0.0


def test_ablation_flags_are_active() -> None:
    direct = build("dapr_direct_unet")
    assert direct.baf_refiner is None

    uniform = build("dapr_baf_uniform_route")
    assert uniform.baf_refiner is not None
    assert uniform.baf_refiner.use_boundary is False

    nonoverlap = build("dapr_baf_nonoverlap")
    assert nonoverlap.baf_refiner is not None
    local = nonoverlap.baf_refiner.local_adapter
    assert local.stride == local.window_size

    no_phase = build("dapr_baf_no_global_phase")
    assert no_phase.fourier_bottleneck.use_phase is False

    no_global_mix = build("dapr_baf_no_global_channel_mix")
    assert no_global_mix.fourier_bottleneck.use_channel_mixing is False

    no_local_mix = build("dapr_baf_no_local_channel_mix")
    assert no_local_mix.baf_refiner is not None
    assert no_local_mix.baf_refiner.local_adapter.use_channel_mixing is False


def test_local_adapter_receives_gradient_on_first_step() -> None:
    torch.manual_seed(19)
    model = build("dapr_baf_unet")
    model.train()
    logits = main_logits(model(torch.randn(1, 3, 32, 32)))
    logits.square().mean().backward()
    refiner = model.baf_refiner
    assert refiner is not None
    gradient = refiner.local_adapter.amplitude_logits.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_boundary_route_is_nonuniform_and_bounded() -> None:
    model = build("dapr_baf_unet").eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))
    route = output["route"]
    assert route.min().item() >= 0.05 - 1.0e-6
    assert route.max().item() <= 1.0 + 1.0e-6
    assert route.std().item() > 0.0


def test_odd_spatial_sizes_are_supported() -> None:
    model = build("dapr_baf_unet").eval()
    with torch.no_grad():
        logits = main_logits(model(torch.randn(1, 3, 35, 37)))
    assert logits.shape == (1, 1, 35, 37)
    assert torch.isfinite(logits).all()


def test_window_chunking_preserves_the_numerical_result() -> None:
    cfg = load_config("dapr_baf_unet")["model"]
    cfg = dict(cfg)
    cfg["channels"] = [4, 8, 16, 32, 64]
    cfg["fourier_init_hw"] = [2, 2]
    cfg["baf_window_size"] = 8
    cfg["baf_stride"] = 4
    cfg["baf_window_chunk_size"] = 2
    small_chunks = build_model("dapr_baf_unet", cfg).eval()

    large_cfg = dict(cfg)
    large_cfg["baf_window_chunk_size"] = 10_000
    large_chunks = build_model("dapr_baf_unet", large_cfg).eval()
    large_chunks.load_state_dict(small_chunks.state_dict(), strict=True)

    x = torch.randn(2, 3, 35, 37)
    with torch.no_grad():
        small_output = main_logits(small_chunks(x))
        large_output = main_logits(large_chunks(x))
    torch.testing.assert_close(small_output, large_output, rtol=1.0e-5, atol=1.0e-6)
