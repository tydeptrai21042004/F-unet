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
ABLATION_MODELS = PROPOSAL_MODELS
