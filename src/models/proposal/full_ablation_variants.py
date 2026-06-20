from __future__ import annotations

"""Permanent component-ablation variants for the two Fourier proposals.

Each registered class changes exactly one architectural factor while reusing
all canonical implementation code.  These variants are normal repository
models; no runtime monkey-patching or generated source files are required.
"""

import torch
import torch.nn.functional as F

from ..registry import register_model
from .apdr_fourier_unet import APDRFourierUNet, _morphological_boundary
from .fourier_unet import PlainFourierUNet


@register_model("plain_fourier_amplitude_only")
class PlainFourierAmplitudeOnly(PlainFourierUNet):
    """Proposal I with global amplitude correction only."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_use_amplitude"] = True
        kwargs["fourier_use_phase"] = False
        super().__init__(*args, **kwargs)


@register_model("plain_fourier_phase_only")
class PlainFourierPhaseOnly(PlainFourierUNet):
    """Proposal I with global phase correction only."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_use_amplitude"] = False
        kwargs["fourier_use_phase"] = True
        super().__init__(*args, **kwargs)


@register_model("plain_fourier_no_channel_mix")
class PlainFourierNoChannelMix(PlainFourierUNet):
    """Proposal I without spectral channel mixing."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_use_channel_mixing"] = False
        super().__init__(*args, **kwargs)


@register_model("plain_fourier_no_residual")
class PlainFourierNoResidual(PlainFourierUNet):
    """Proposal I without the residual spatial bypass."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_residual"] = False
        # The canonical bottleneck forbids zero initialization when the
        # residual bypass is absent, so this dependent switch changes too.
        kwargs["fourier_zero_init_output"] = False
        super().__init__(*args, **kwargs)


class _ConfigurableAPDRFourierUNet(APDRFourierUNet):
    """APDR implementation with explicit one-factor ablation switches."""

    def __init__(
        self,
        *args,
        ablate_disagreement: bool = False,
        ablate_uncertainty: bool = False,
        ablate_boundary: bool = False,
        ablate_context: bool = False,
        uniform_route: bool = False,
        local_use_amplitude: bool = True,
        local_use_phase: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ablate_disagreement = bool(ablate_disagreement)
        self.ablate_uncertainty = bool(ablate_uncertainty)
        self.ablate_boundary = bool(ablate_boundary)
        self.ablate_context = bool(ablate_context)
        self.uniform_route = bool(uniform_route)
        self.local_use_amplitude = bool(local_use_amplitude)
        self.local_use_phase = bool(local_use_phase)

        local = self.apdr_adapter.local_adapter
        if not self.local_use_amplitude:
            local.amplitude_scale = 0.0
        if not self.local_use_phase:
            local.phase_max = 0.0

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        decoded, disagreement = self._plain_forward_features(
            x,
            return_disagreement=True,
        )
        assert disagreement is not None

        baseline_logits = self.seg_head(decoded)
        adapter = self.apdr_adapter

        if adapter.detach_disagreement:
            disagreement = disagreement.detach()
        disagreement = F.interpolate(
            disagreement,
            size=decoded.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        probability = torch.sigmoid(baseline_logits)
        uncertainty = 4.0 * probability * (1.0 - probability)
        boundary = _morphological_boundary(probability)
        context = adapter.context(decoded)

        if self.ablate_disagreement:
            disagreement = torch.zeros_like(disagreement)
        if self.ablate_uncertainty:
            uncertainty = torch.zeros_like(uncertainty)
        if self.ablate_boundary:
            boundary = torch.zeros_like(boundary)
        if self.ablate_context:
            context = torch.zeros_like(context)

        if self.uniform_route:
            route = torch.ones_like(uncertainty)
        else:
            route_logits = adapter.router(
                torch.cat(
                    [context, disagreement, uncertainty, boundary],
                    dim=1,
                )
            )
            route = torch.sigmoid(route_logits)
            route = adapter.routing_floor + (1.0 - adapter.routing_floor) * route

        local_residual = adapter.local_adapter(decoded)
        beta = adapter.effective_beta()
        refined = decoded + beta * route * local_residual
        main_logits = self.seg_head(refined)

        with torch.no_grad():
            adapter._last_stats = {
                "apdr/beta": float(beta.detach().item()),
                "apdr/route_mean": float(route.mean().item()),
                "apdr/route_min": float(route.min().item()),
                "apdr/route_max": float(route.max().item()),
                "apdr/disagreement_mean": float(disagreement.mean().item()),
                "apdr/uncertainty_mean": float(uncertainty.mean().item()),
                "apdr/boundary_mean": float(boundary.mean().item()),
            }

        return {
            "main": main_logits,
            "baseline_logits": baseline_logits,
            "route": route,
            "disagreement": disagreement,
            "uncertainty": uncertainty,
            "boundary_evidence": boundary,
            "local_residual": local_residual,
            "beta": beta,
        }


@register_model("apdr_uniform_route")
class APDRUniformRoute(_ConfigurableAPDRFourierUNet):
    """APDR local residual with no learned routing map."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, uniform_route=True, **kwargs)


@register_model("apdr_no_disagreement")
class APDRNoDisagreement(_ConfigurableAPDRFourierUNet):
    """APDR without amplitude--phase disagreement evidence."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, ablate_disagreement=True, **kwargs)


@register_model("apdr_no_uncertainty")
class APDRNoUncertainty(_ConfigurableAPDRFourierUNet):
    """APDR without predictive-uncertainty evidence."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, ablate_uncertainty=True, **kwargs)


@register_model("apdr_no_boundary")
class APDRNoBoundary(_ConfigurableAPDRFourierUNet):
    """APDR without boundary evidence."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, ablate_boundary=True, **kwargs)


@register_model("apdr_no_context")
class APDRNoContext(_ConfigurableAPDRFourierUNet):
    """APDR without decoder-context evidence."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, ablate_context=True, **kwargs)


@register_model("apdr_local_amplitude_only")
class APDRLocalAmplitudeOnly(_ConfigurableAPDRFourierUNet):
    """APDR whose local adapter uses amplitude correction only."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            local_use_amplitude=True,
            local_use_phase=False,
            **kwargs,
        )


@register_model("apdr_local_phase_only")
class APDRLocalPhaseOnly(_ConfigurableAPDRFourierUNet):
    """APDR whose local adapter uses phase correction only."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            local_use_amplitude=False,
            local_use_phase=True,
            **kwargs,
        )


__all__ = [
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
