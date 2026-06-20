from __future__ import annotations

"""Permanent one-factor ablations for DAPR-BAF U-Net."""

from ..registry import register_model
from .dapr_baf_unet import DAPRBAFUNet


@register_model("dapr_direct_unet")
class DAPRDirectUNet(DAPRBAFUNet):
    """Direct global amplitude--phase reconstruction without local refinement."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["baf_enabled"] = False
        super().__init__(*args, **kwargs)


@register_model("dapr_baf_uniform_route")
class DAPRBAFUniformRoute(DAPRBAFUNet):
    """Local amplitude refinement applied uniformly instead of at boundaries."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["baf_use_boundary"] = False
        super().__init__(*args, **kwargs)


@register_model("dapr_baf_nonoverlap")
class DAPRBAFNonOverlapping(DAPRBAFUNet):
    """Local amplitude refinement with non-overlapping windows."""

    def __init__(self, *args, **kwargs) -> None:
        window_size = int(kwargs.get("baf_window_size", 16))
        kwargs["baf_stride"] = window_size
        super().__init__(*args, **kwargs)


@register_model("dapr_baf_no_global_phase")
class DAPRBAFNoGlobalPhase(DAPRBAFUNet):
    """Direct global reconstruction with amplitude modulation only."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_use_phase"] = False
        super().__init__(*args, **kwargs)


@register_model("dapr_baf_no_global_channel_mix")
class DAPRBAFNoGlobalChannelMix(DAPRBAFUNet):
    """DAPR-BAF without the global spectral channel mixer."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_use_channel_mixing"] = False
        super().__init__(*args, **kwargs)


@register_model("dapr_baf_no_local_channel_mix")
class DAPRBAFNoLocalChannelMix(DAPRBAFUNet):
    """DAPR-BAF without the local spectral channel mixer."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["baf_use_channel_mixing"] = False
        super().__init__(*args, **kwargs)


__all__ = [
    "DAPRDirectUNet",
    "DAPRBAFUniformRoute",
    "DAPRBAFNonOverlapping",
    "DAPRBAFNoGlobalPhase",
    "DAPRBAFNoGlobalChannelMix",
    "DAPRBAFNoLocalChannelMix",
]
