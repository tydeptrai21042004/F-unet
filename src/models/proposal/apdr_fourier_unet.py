from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fourier_unet import _PlainFourierUNetBase
from ..registry import register_model


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(16, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _morphological_boundary(probability: torch.Tensor) -> torch.Tensor:
    maximum = F.max_pool2d(probability, kernel_size=3, stride=1, padding=1)
    minimum = -F.max_pool2d(-probability, kernel_size=3, stride=1, padding=1)
    return (maximum - minimum).clamp(0.0, 1.0)


class WindowedFourierResidualAdapter(nn.Module):
    """Non-overlapping local Fourier residual adapter.

    Only middle/high-frequency responses are learned.  The module itself is
    non-zero at initialization so the outer zero-initialized residual scalar
    receives a useful gradient on the first optimization step.
    """

    def __init__(
        self,
        channels: int,
        window_size: int = 16,
        expansion: float = 1.0,
        amplitude_scale: float = 0.20,
        phase_max: float = math.pi / 4.0,
        highpass_start: float = 0.20,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if window_size < 4:
            raise ValueError("window_size must be at least 4")
        if expansion <= 0:
            raise ValueError("expansion must be positive")
        if amplitude_scale < 0 or phase_max < 0:
            raise ValueError("spectral limits must be non-negative")
        if not 0.0 <= highpass_start < 1.0:
            raise ValueError("highpass_start must lie in [0, 1)")

        hidden = max(channels, int(round(channels * float(expansion))))
        self.channels = int(channels)
        self.hidden_channels = int(hidden)
        self.window_size = int(window_size)
        self.amplitude_scale = float(amplitude_scale)
        self.phase_max = float(phase_max)

        self.pre = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            _group_norm(hidden),
            nn.GELU(),
        )
        wf = self.window_size // 2 + 1
        self.amplitude_logits = nn.Parameter(torch.empty(1, hidden, self.window_size, wf))
        self.phase_logits = nn.Parameter(torch.empty(1, hidden, self.window_size, wf))
        self.spectral_channel_mixer = nn.Conv2d(hidden, hidden, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.post = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

        fy = torch.fft.fftfreq(self.window_size)
        fx = torch.fft.rfftfreq(self.window_size)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        radius = radius / radius.max().clamp_min(1.0e-6)
        mask = torch.sigmoid((radius - float(highpass_start)) * 12.0)
        self.register_buffer("frequency_mask", mask.view(1, 1, self.window_size, wf), persistent=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        conv = self.pre[0]
        assert isinstance(conv, nn.Conv2d)
        nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.normal_(self.amplitude_logits, mean=0.0, std=2.0e-3)
        nn.init.normal_(self.phase_logits, mean=0.0, std=2.0e-3)
        nn.init.zeros_(self.spectral_channel_mixer.weight)
        eye = torch.eye(self.hidden_channels, dtype=self.spectral_channel_mixer.weight.dtype)
        self.spectral_channel_mixer.weight.data[:, :, 0, 0].copy_(eye)
        nn.init.kaiming_normal_(self.post.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.post.bias)

    def _window_partition(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        batch, channels, height, width = x.shape
        ws = self.window_size
        pad_h = (ws - height % ws) % ws
        pad_w = (ws - width % ws) % ws
        if pad_h or pad_w:
            mode = "reflect" if height > pad_h and width > pad_w else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        padded_h, padded_w = x.shape[-2:]
        nh, nw = padded_h // ws, padded_w // ws
        windows = (
            x.view(batch, channels, nh, ws, nw, ws)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch * nh * nw, channels, ws, ws)
        )
        return windows, (height, width, nh, nw)

    def _window_reverse(
        self,
        windows: torch.Tensor,
        meta: tuple[int, int, int, int],
        batch: int,
    ) -> torch.Tensor:
        height, width, nh, nw = meta
        ws = self.window_size
        channels = windows.shape[1]
        x = (
            windows.view(batch, nh, nw, channels, ws, ws)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, nh * ws, nw * ws)
        )
        return x[..., :height, :width]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre(x)
        batch = z.shape[0]
        windows, meta = self._window_partition(z)
        spectrum = torch.fft.rfft2(windows, dim=(-2, -1), norm="ortho")
        mask = self.frequency_mask.to(dtype=windows.dtype)
        gain = torch.exp(self.amplitude_scale * torch.tanh(self.amplitude_logits) * mask)
        phase = self.phase_max * torch.tanh(self.phase_logits) * mask
        transformed = spectrum * torch.polar(gain, phase)
        transformed = torch.complex(
            self.spectral_channel_mixer(transformed.real),
            self.spectral_channel_mixer(transformed.imag),
        )
        residual_windows = torch.fft.irfft2(
            transformed - spectrum,
            s=(self.window_size, self.window_size),
            dim=(-2, -1),
            norm="ortho",
        )
        residual = self._window_reverse(residual_windows, meta, batch)
        return self.post(self.dropout(residual))

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        gain = torch.exp(
            self.amplitude_scale
            * torch.tanh(self.amplitude_logits)
            * self.frequency_mask
        )
        phase = self.phase_max * torch.tanh(self.phase_logits) * self.frequency_mask
        return {
            "apdr/local_gain_mean": float(gain.mean().item()),
            "apdr/local_phase_abs_mean": float(phase.abs().mean().item()),
        }


class APDRResidualAdapter(nn.Module):
    """Amplitude--phase disagreement-routed residual adapter."""

    def __init__(
        self,
        channels: int,
        window_size: int = 16,
        expansion: float = 1.0,
        amplitude_scale: float = 0.20,
        phase_max: float = math.pi / 4.0,
        highpass_start: float = 0.20,
        dropout: float = 0.05,
        beta_max: float = 0.25,
        routing_floor: float = 0.05,
        detach_disagreement: bool = False,
    ) -> None:
        super().__init__()
        if beta_max <= 0:
            raise ValueError("beta_max must be positive")
        if not 0.0 <= routing_floor < 1.0:
            raise ValueError("routing_floor must lie in [0, 1)")
        context_channels = max(8, channels // 4)
        self.beta_max = float(beta_max)
        self.routing_floor = float(routing_floor)
        self.detach_disagreement = bool(detach_disagreement)
        self.schedule_scale = 1.0

        self.context = nn.Sequential(
            nn.Conv2d(channels, context_channels, kernel_size=1, bias=False),
            _group_norm(context_channels),
            nn.GELU(),
        )
        self.router = nn.Sequential(
            nn.Conv2d(context_channels + 3, context_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(context_channels),
            nn.GELU(),
            nn.Conv2d(context_channels, 1, kernel_size=1, bias=True),
        )
        self.local_adapter = WindowedFourierResidualAdapter(
            channels=channels,
            window_size=window_size,
            expansion=expansion,
            amplitude_scale=amplitude_scale,
            phase_max=phase_max,
            highpass_start=highpass_start,
            dropout=dropout,
        )
        self.beta_raw = nn.Parameter(torch.zeros(()))
        self._last_stats: dict[str, float] = {}
        self.reset_router()

    def reset_router(self) -> None:
        for module in self.context.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        for module in self.router.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.beta_raw)

    def set_schedule_scale(self, value: float) -> None:
        self.schedule_scale = float(min(max(value, 0.0), 1.0))

    def effective_beta(self) -> torch.Tensor:
        return self.schedule_scale * self.beta_max * torch.tanh(self.beta_raw)

    def forward(
        self,
        decoded: torch.Tensor,
        baseline_logits: torch.Tensor,
        disagreement: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.detach_disagreement:
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
        route_logits = self.router(
            torch.cat([self.context(decoded), disagreement, uncertainty, boundary], dim=1)
        )
        route = torch.sigmoid(route_logits)
        route = self.routing_floor + (1.0 - self.routing_floor) * route
        local_residual = self.local_adapter(decoded)
        beta = self.effective_beta()
        refined = decoded + beta * route * local_residual

        with torch.no_grad():
            self._last_stats = {
                "apdr/beta": float(beta.detach().item()),
                "apdr/route_mean": float(route.mean().item()),
                "apdr/route_min": float(route.min().item()),
                "apdr/route_max": float(route.max().item()),
                "apdr/disagreement_mean": float(disagreement.mean().item()),
                "apdr/uncertainty_mean": float(uncertainty.mean().item()),
                "apdr/boundary_mean": float(boundary.mean().item()),
            }
        return refined, {
            "route": route,
            "disagreement": disagreement,
            "uncertainty": uncertainty,
            "boundary_evidence": boundary,
            "local_residual": local_residual,
            "beta": beta,
        }

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        values = dict(self._last_stats)
        values.update(self.local_adapter.diagnostics())
        return values


@register_model("apdr_fourier_unet")
class APDRFourierUNet(_PlainFourierUNetBase):
    """Amplitude--Phase Disagreement-Routed Residual Fourier U-Net.

    The full Plain Fourier U-Net is retained as an immutable main path.  APDR
    adds a local residual Fourier adapter after decoding.  Its scalar gate is
    initialized to exactly zero, so the network output is identical to Plain
    Fourier U-Net before learning the new residual branch.
    """

    def __init__(
        self,
        *args,
        apdr_window_size: int = 16,
        apdr_expansion: float = 1.0,
        apdr_amplitude_scale: float = 0.20,
        apdr_phase_max: float = math.pi / 4.0,
        apdr_highpass_start: float = 0.20,
        apdr_dropout: float = 0.05,
        apdr_beta_max: float = 0.25,
        apdr_routing_floor: float = 0.05,
        apdr_detach_disagreement: bool = False,
        apdr_warmup_epochs: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        decoder_channels = int(self.seg_head.in_channels)
        self.apdr_warmup_epochs = int(apdr_warmup_epochs)
        self.apdr_adapter = APDRResidualAdapter(
            channels=decoder_channels,
            window_size=apdr_window_size,
            expansion=apdr_expansion,
            amplitude_scale=apdr_amplitude_scale,
            phase_max=apdr_phase_max,
            highpass_start=apdr_highpass_start,
            dropout=apdr_dropout,
            beta_max=apdr_beta_max,
            routing_floor=apdr_routing_floor,
            detach_disagreement=apdr_detach_disagreement,
        )
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        if not hasattr(self, "apdr_adapter"):
            return
        if self.apdr_warmup_epochs <= 0:
            scale = 1.0
        else:
            scale = min(max(float(epoch), 0.0) / float(self.apdr_warmup_epochs), 1.0)
        self.apdr_adapter.set_schedule_scale(scale)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        decoded, disagreement = self._plain_forward_features(x, return_disagreement=True)
        assert disagreement is not None
        baseline_logits = self.seg_head(decoded)
        refined, evidence = self.apdr_adapter(decoded, baseline_logits, disagreement)
        main_logits = self.seg_head(refined)
        return {
            "main": main_logits,
            "baseline_logits": baseline_logits,
            **evidence,
        }

    def diagnostic_metrics(self) -> dict[str, float]:
        values = super().diagnostic_metrics()
        values.update(self.apdr_adapter.diagnostics())
        return values


__all__ = [
    "WindowedFourierResidualAdapter",
    "APDRResidualAdapter",
    "APDRFourierUNet",
]
