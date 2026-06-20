"""Canonical scope and labels for the DAPR-BAF component ablation."""

from __future__ import annotations

DAPR_BAF_ABLATION_MODELS = [
    "dapr_direct_unet",
    "dapr_baf_uniform_route",
    "dapr_baf_nonoverlap",
    "dapr_baf_no_global_phase",
    "dapr_baf_no_global_channel_mix",
    "dapr_baf_no_local_channel_mix",
    "dapr_baf_unet",
]

DISPLAY_NAMES = {
    "dapr_direct_unet": "Direct reconstruction only",
    "dapr_baf_uniform_route": "No boundary routing",
    "dapr_baf_nonoverlap": "Non-overlapping local windows",
    "dapr_baf_no_global_phase": "No global phase modulation",
    "dapr_baf_no_global_channel_mix": "No global channel mixing",
    "dapr_baf_no_local_channel_mix": "No local channel mixing",
    "dapr_baf_unet": "Full DAPR-BAF U-Net",
}

COMPONENT_COMPARISONS = [
    (model, "dapr_baf_unet")
    for model in DAPR_BAF_ABLATION_MODELS
    if model != "dapr_baf_unet"
]

DEFAULT_SEEDS = [42, 1, 2]
DEFAULT_MODELS_CSV = ",".join(DAPR_BAF_ABLATION_MODELS)
