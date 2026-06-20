"""Canonical model lists used by all benchmark entrypoints."""

BASELINE_MODELS = [
    "unet",
    "unetpp",
    "attention_unet",
    "pranet",
    "acsnet",
    "hardnet_mseg",
    "cfanet",
    "polyp_pvt",
    "caranet",
    "hsnet",
    "resunetpp",
]

PROPOSAL_MODELS = [
    "plain_fourier_unet",
    "apdr_fourier_unet",
]

FAIR_MODELS = BASELINE_MODELS + PROPOSAL_MODELS
BASELINE_PROPOSAL_MODELS = FAIR_MODELS
ABLATION_MODELS = PROPOSAL_MODELS

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
