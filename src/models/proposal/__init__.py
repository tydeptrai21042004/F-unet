from .fourier_unet import FourierSpectralBottleneck, PlainFourierUNet
from .apdr_fourier_unet import (
    APDRFourierUNet,
    APDRResidualAdapter,
    WindowedFourierResidualAdapter,
)

__all__ = [
    "FourierSpectralBottleneck",
    "PlainFourierUNet",
    "WindowedFourierResidualAdapter",
    "APDRResidualAdapter",
    "APDRFourierUNet",
    "PlainFourierAmplitudeOnly",
    "PlainFourierPhaseOnly",
    "PlainFourierNoChannelMix",
    "PlainFourierNoResidual",
    "APDRUniformRoute",
    "APDRNoDisagreement",
    "APDRNoUncertainty",
    "APDRNoBoundary",
    "APDRNoContext",
    "APDRLocalAmplitudeOnly",
    "APDRLocalPhaseOnly",
]

from .full_ablation_variants import (
    APDRLocalAmplitudeOnly,
    APDRLocalPhaseOnly,
    APDRNoBoundary,
    APDRNoContext,
    APDRNoDisagreement,
    APDRNoUncertainty,
    APDRUniformRoute,
    PlainFourierAmplitudeOnly,
    PlainFourierNoChannelMix,
    PlainFourierNoResidual,
    PlainFourierPhaseOnly,
)
