from __future__ import annotations

import ast
from pathlib import Path

import yaml

from src.models.registry import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
BASELINES = {
    "unet", "unetpp", "attention_unet", "pranet", "acsnet",
    "hardnet_mseg", "cfanet", "polyp_pvt", "caranet", "hsnet",
    "resunetpp",
}
PROPOSALS = {"plain_fourier_unet", "apdr_fourier_unet", "dapr_baf_unet"}
FULL_ABLATION = {
    "unet", "plain_fourier_amplitude_only", "plain_fourier_phase_only",
    "plain_fourier_no_channel_mix", "plain_fourier_no_residual",
    "plain_fourier_unet", "apdr_uniform_route",
    "apdr_no_disagreement", "apdr_no_uncertainty",
    "apdr_no_boundary", "apdr_no_context",
    "apdr_local_amplitude_only", "apdr_local_phase_only",
    "apdr_fourier_unet",
}
CANONICAL = BASELINES | PROPOSALS
DAPR_BAF_ABLATION = {
    "dapr_direct_unet", "dapr_baf_uniform_route",
    "dapr_baf_nonoverlap", "dapr_baf_no_global_phase",
    "dapr_baf_no_global_channel_mix", "dapr_baf_no_local_channel_mix",
    "dapr_baf_unet",
}
REGISTRY_MODELS = CANONICAL | FULL_ABLATION | DAPR_BAF_ABLATION


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_registry_contains_only_requested_models() -> None:
    assert set(MODEL_REGISTRY) == REGISTRY_MODELS


def test_config_directories_have_exact_scope() -> None:
    assert {p.name for p in CONFIGS.iterdir() if p.is_dir()} == {
        "official_faithful", "fair", "ablation", "full_component_ablation",
        "dapr_baf_ablation"
    }
    assert {p.stem for p in (CONFIGS / "official_faithful").glob("*.yaml")} == BASELINES
    assert {p.stem for p in (CONFIGS / "fair").glob("*.yaml")} == CANONICAL
    assert {p.stem for p in (CONFIGS / "ablation").glob("*.yaml")} == PROPOSALS
    assert {p.stem for p in (CONFIGS / "full_component_ablation").glob("*.yaml")} == FULL_ABLATION
    assert {p.stem for p in (CONFIGS / "dapr_baf_ablation").glob("*.yaml")} == DAPR_BAF_ABLATION


def test_all_fair_configs_share_the_same_training_and_evaluation_protocol() -> None:
    configs = {p.stem: load(p) for p in (CONFIGS / "fair").glob("*.yaml")}
    reference = configs["plain_fourier_unet"]
    shared_paths = [
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
    for section, key in shared_paths:
        expected = reference[section][key]
        for name, cfg in configs.items():
            assert cfg[section][key] == expected, (name, section, key)


def test_apdr_changes_only_apdr_specific_model_keys() -> None:
    plain = load(CONFIGS / "ablation" / "plain_fourier_unet.yaml")
    apdr = load(CONFIGS / "ablation" / "apdr_fourier_unet.yaml")
    assert plain["data"] == apdr["data"]
    assert plain["train"] == apdr["train"]
    assert plain["eval"] == apdr["eval"]
    plain_model = plain["model"]
    apdr_model = apdr["model"]
    for key, value in plain_model.items():
        if key == "name":
            continue
        assert apdr_model[key] == value, key
    extra = set(apdr_model) - set(plain_model)
    assert extra
    assert all(key.startswith("apdr_") for key in extra)


def test_ablation_runner_contains_exactly_two_models() -> None:
    tree = ast.parse((ROOT / "scripts" / "run_apdr_ablation.py").read_text(encoding="utf-8"))
    values = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert "plain_fourier_unet,apdr_fourier_unet" in values
    dapr_tree = ast.parse((ROOT / "scripts" / "run_dapr_baf_ablation.py").read_text(encoding="utf-8"))
    dapr_values = [
        node.value for node in ast.walk(dapr_tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert "configs/dapr_baf_ablation" in dapr_values


def test_removed_proposal_names_are_absent_from_active_code() -> None:
    forbidden = [
        "proposal_apf_unet", "proposal_urf_unet", "proposal_fourier_unet",
        "fourier_unet_bounded", "fourier_unet_amplitude_only",
        "fourier_unet_phase_only", "fourier_unet_no_channel_mix",
        "fourier_unet_no_residual", "fourier_unet_at_encoder1",
        "csca_unet",
    ]
    roots = [ROOT / "src", ROOT / "configs", ROOT / "scripts", ROOT / "tools"]
    hits: list[tuple[str, str]] = []
    for base in roots:
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml", ".sh"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                if token in text:
                    hits.append((str(path.relative_to(ROOT)), token))
    assert hits == []


def test_manuscript_baseline_fragment_contains_exact_requested_rows() -> None:
    text = (ROOT / "docs" / "manuscript_baselines.tex").read_text(encoding="utf-8")
    required = [
        "U-Net", "U-Net++", "Attention U-Net", "PraNet", "ACSNet",
        "HarDNet-MSEG", "CFANet", "Polyp-PVT", "CaraNet", "HSNet",
        "ResUNet++",
    ]
    for name in required:
        assert name in text
    expected_values = [
        "$0.0513 \\pm 0.0090$", "$0.1675 \\pm 0.0193$",
        "$0.0537 \\pm 0.0085$", "$0.1724 \\pm 0.0170$",
        "$0.0559 \\pm 0.0024$", "$0.1786 \\pm 0.0075$",
        "$0.0525 \\pm 0.0060$", "$0.1704 \\pm 0.0236$",
        "$0.0612 \\pm 0.0039$", "$0.1749 \\pm 0.0040$",
        "$0.0428 \\pm 0.0043$", "$0.1242 \\pm 0.0024$",
        "$0.0521 \\pm 0.0044$", "$0.1780 \\pm 0.0049$",
        "$0.0591 \\pm 0.0072$", "$0.1695 \\pm 0.0081$",
        "$0.0521 \\pm 0.0077$", "$0.1599 \\pm 0.0198$",
        "$0.0477 \\pm 0.0031$", "$0.1359 \\pm 0.0075$",
        "$0.0475 \\pm 0.0053$", "$0.1477 \\pm 0.0104$",
    ]
    for value in expected_values:
        assert value in text
    assert text.count("\\\\") == 11



def test_ablation_documentation_frames_both_spectral_models_as_proposals() -> None:
    readme = (ROOT / "configs" / "ablation" / "README.md").read_text(encoding="utf-8")
    assert "Proposal I: Plain Fourier U-Net" in readme
    assert "Proposal II: APDR-Fourier U-Net" in readme
    assert "Proposal III: DAPR-BAF U-Net" in readme
    assert "spectral baseline" not in readme.lower()


def test_isic2018_support_and_manuscript_paragraph_are_removed() -> None:
    active_roots = [
        ROOT / "src",
        ROOT / "scripts",
        ROOT / "configs",
        ROOT / "docs",
        ROOT / "README.md",
        ROOT / "APDR_FOURIER_UNET_GUIDE.md",
    ]
    forbidden = ("isic2018", "ISIC2018", "ISIC 2018", "ISIC-2018")
    hits: list[tuple[str, str]] = []
    for root in active_roots:
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".tex", ".yaml", ".yml", ".sh"}:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                if token in content:
                    hits.append((str(path.relative_to(ROOT)), token))
    assert hits == []

    manuscript = (ROOT / "docs" / "manuscript_two_proposals.tex").read_text(encoding="utf-8")
    assert r"\paragraph{ISBI-2012 challenge.}" in manuscript
    assert r"\paragraph{ISBI-2018-challenge.}" not in manuscript
