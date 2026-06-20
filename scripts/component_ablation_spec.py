"""Canonical names, labels, groups and comparisons for full component ablation."""

from __future__ import annotations

FULL_COMPONENT_ABLATION_MODELS = [
    "unet",
    "plain_fourier_amplitude_only",
    "plain_fourier_phase_only",
    "plain_fourier_no_channel_mix",
    "plain_fourier_no_residual",
    "plain_fourier_unet",
    "apdr_uniform_route",
    "apdr_no_disagreement",
    "apdr_no_uncertainty",
    "apdr_no_boundary",
    "apdr_no_context",
    "apdr_local_amplitude_only",
    "apdr_local_phase_only",
    "apdr_fourier_unet",
]

DISPLAY_NAMES = {
    "unet": "U-Net backbone",
    "plain_fourier_amplitude_only": "Proposal I: amplitude only",
    "plain_fourier_phase_only": "Proposal I: phase only",
    "plain_fourier_no_channel_mix": "Proposal I: no channel mixing",
    "plain_fourier_no_residual": "Proposal I: no residual bypass",
    "plain_fourier_unet": "Proposal I: Plain Fourier U-Net",
    "apdr_uniform_route": "Proposal II: uniform route",
    "apdr_no_disagreement": "Proposal II: no AP disagreement",
    "apdr_no_uncertainty": "Proposal II: no uncertainty",
    "apdr_no_boundary": "Proposal II: no boundary evidence",
    "apdr_no_context": "Proposal II: no decoder context",
    "apdr_local_amplitude_only": "Proposal II: local amplitude only",
    "apdr_local_phase_only": "Proposal II: local phase only",
    "apdr_fourier_unet": "Proposal II: full APDR-Fourier U-Net",
}

COMPONENT_COMPARISONS = [
    ("plain_fourier_amplitude_only", "plain_fourier_unet"),
    ("plain_fourier_phase_only", "plain_fourier_unet"),
    ("plain_fourier_no_channel_mix", "plain_fourier_unet"),
    ("plain_fourier_no_residual", "plain_fourier_unet"),
    ("apdr_uniform_route", "apdr_fourier_unet"),
    ("apdr_no_disagreement", "apdr_fourier_unet"),
    ("apdr_no_uncertainty", "apdr_fourier_unet"),
    ("apdr_no_boundary", "apdr_fourier_unet"),
    ("apdr_no_context", "apdr_fourier_unet"),
    ("apdr_local_amplitude_only", "apdr_fourier_unet"),
    ("apdr_local_phase_only", "apdr_fourier_unet"),
    ("plain_fourier_unet", "apdr_fourier_unet"),
]

DEFAULT_SEEDS = [42, 1, 2]
DEFAULT_MODELS_CSV = ",".join(FULL_COMPONENT_ABLATION_MODELS)
