from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from .apdr_fourier_unet import _group_norm, _morphological_boundary
from .fourier_unet import _PlainFourierUNetBase


class OverlappingAmplitudeResidualAdapter(nn.Module):
    """Overlapping local amplitude-only Fourier residual adapter.

    Windows are processed with a learnable middle/high-frequency amplitude
    response and recombined by normalized raised-cosine overlap-add. Local
    phase is preserved exactly.
    """

    def __init__(
        self,
        channels: int,
        window_size: int = 16,
        stride: int = 8,
        expansion: float = 1.0,
        amplitude_scale: float = 0.20,
        highpass_start: float = 0.20,
        dropout: float = 0.05,
        use_channel_mixing: bool = True,
        synthesis_floor: float = 0.05,
        window_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if window_size < 4:
            raise ValueError("window_size must be at least 4")
        if stride <= 0 or stride > window_size:
            raise ValueError("stride must lie in [1, window_size]")
        if expansion <= 0:
            raise ValueError("expansion must be positive")
        if amplitude_scale < 0:
            raise ValueError("amplitude_scale must be non-negative")
        if not 0.0 <= highpass_start < 1.0:
            raise ValueError("highpass_start must lie in [0, 1)")
        if not 0.0 < synthesis_floor <= 1.0:
            raise ValueError("synthesis_floor must lie in (0, 1]")
        if window_chunk_size <= 0:
            raise ValueError("window_chunk_size must be positive")

        hidden = max(int(channels), int(round(channels * float(expansion))))
        self.channels = int(channels)
        self.hidden_channels = int(hidden)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.amplitude_scale = float(amplitude_scale)
        self.use_channel_mixing = bool(use_channel_mixing)
        self.window_chunk_size = int(window_chunk_size)

        self.pre = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            _group_norm(hidden),
            nn.GELU(),
        )
        wf = self.window_size // 2 + 1
        self.amplitude_logits = nn.Parameter(
            torch.empty(1, hidden, self.window_size, wf)
        )
        if self.use_channel_mixing:
            self.spectral_channel_mixer: nn.Module = nn.Conv2d(
                hidden, hidden, kernel_size=1, bias=False
            )
        else:
            self.spectral_channel_mixer = nn.Identity()
        self.dropout = (
            nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        )
        self.post = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

        fy = torch.fft.fftfreq(self.window_size)
        fx = torch.fft.rfftfreq(self.window_size)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        radius = radius / radius.max().clamp_min(1.0e-6)
        frequency_mask = torch.sigmoid(
            (radius - float(highpass_start)) * 12.0
        )
        self.register_buffer(
            "frequency_mask",
            frequency_mask.view(1, 1, self.window_size, wf),
            persistent=True,
        )

        one_d = torch.hann_window(self.window_size, periodic=False)
        synthesis = torch.outer(one_d, one_d)
        synthesis = synthesis / synthesis.max().clamp_min(1.0e-6)
        synthesis = float(synthesis_floor) + (1.0 - float(synthesis_floor)) * synthesis
        self.register_buffer(
            "synthesis_window",
            synthesis.view(1, 1, self.window_size, self.window_size),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        conv = self.pre[0]
        assert isinstance(conv, nn.Conv2d)
        nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.normal_(self.amplitude_logits, mean=0.0, std=2.0e-3)
        if isinstance(self.spectral_channel_mixer, nn.Conv2d):
            nn.init.zeros_(self.spectral_channel_mixer.weight)
            eye = torch.eye(
                self.hidden_channels,
                dtype=self.spectral_channel_mixer.weight.dtype,
            )
            self.spectral_channel_mixer.weight.data[:, :, 0, 0].copy_(eye)
        nn.init.kaiming_normal_(self.post.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.post.bias)

    def _padding(self, height: int, width: int) -> tuple[int, int, int, int]:
        ws = self.window_size
        stride = self.stride

        def padded_size(size: int) -> int:
            if size <= ws:
                return ws
            steps = math.ceil((size - ws) / stride)
            return ws + steps * stride

        padded_h = padded_size(height)
        padded_w = padded_size(width)
        return padded_h - height, padded_w - width, padded_h, padded_w

    def _process_window_chunk(self, windows: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft2(windows, dim=(-2, -1), norm="ortho")
        mask = self.frequency_mask.to(dtype=windows.dtype, device=windows.device)
        gain = torch.exp(
            self.amplitude_scale * torch.tanh(self.amplitude_logits) * mask
        )
        transformed = spectrum * gain
        if isinstance(self.spectral_channel_mixer, nn.Conv2d):
            transformed = torch.complex(
                self.spectral_channel_mixer(transformed.real),
                self.spectral_channel_mixer(transformed.imag),
            )
        return torch.fft.irfft2(
            transformed - spectrum,
            s=(self.window_size, self.window_size),
            dim=(-2, -1),
            norm="ortho",
        )

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = x.shape
        pad_h, pad_w, padded_h, padded_w = self._padding(height, width)
        if pad_h or pad_w:
            mode = "reflect" if height > pad_h and width > pad_w else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

        columns = F.unfold(
            x,
            kernel_size=self.window_size,
            stride=self.stride,
        )
        num_windows = int(columns.shape[-1])
        weight = self.synthesis_window.to(dtype=x.dtype, device=x.device)
        residual_column_chunks: list[torch.Tensor] = []
        for start in range(0, num_windows, self.window_chunk_size):
            stop = min(start + self.window_chunk_size, num_windows)
            chunk_columns = columns[:, :, start:stop]
            chunk_count = stop - start
            windows = (
                chunk_columns.transpose(1, 2)
                .reshape(
                    chunk_count,
                    channels,
                    self.window_size,
                    self.window_size,
                )
            )
            residual_windows = self._process_window_chunk(windows) * weight
            residual_column_chunks.append(
                residual_windows.reshape(
                    1,
                    chunk_count,
                    channels * self.window_size * self.window_size,
                )
                .transpose(1, 2)
                .contiguous()
            )

        residual_columns = torch.cat(residual_column_chunks, dim=-1)
        output = F.fold(
            residual_columns,
            output_size=(padded_h, padded_w),
            kernel_size=self.window_size,
            stride=self.stride,
        )
        norm_columns = (
            weight.reshape(1, self.window_size * self.window_size, 1)
            .expand(1, self.window_size * self.window_size, num_windows)
            .contiguous()
        )
        normalizer = F.fold(
            norm_columns,
            output_size=(padded_h, padded_w),
            kernel_size=self.window_size,
            stride=self.stride,
        )
        output = output / normalizer.clamp_min(1.0e-6)
        return output[..., :height, :width]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre(x)
        residual = torch.cat(
            [self._forward_single(z[index : index + 1]) for index in range(z.shape[0])],
            dim=0,
        )
        return self.post(self.dropout(residual))

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        gain = torch.exp(
            self.amplitude_scale
            * torch.tanh(self.amplitude_logits)
            * self.frequency_mask
        )
        return {
            "dapr_baf/local_gain_mean": float(gain.mean().item()),
            "dapr_baf/local_gain_min": float(gain.min().item()),
            "dapr_baf/local_gain_max": float(gain.max().item()),
            "dapr_baf/window_stride": float(self.stride),
        }


class BoundaryGuidedAmplitudeRefiner(nn.Module):
    """Prediction-derived boundary gate for local amplitude refinement."""

    def __init__(
        self,
        channels: int,
        window_size: int = 16,
        stride: int = 8,
        expansion: float = 1.0,
        amplitude_scale: float = 0.20,
        highpass_start: float = 0.20,
        dropout: float = 0.05,
        beta_max: float = 0.25,
        beta_init: float = 0.02,
        routing_floor: float = 0.05,
        use_boundary: bool = True,
        detach_boundary: bool = False,
        use_channel_mixing: bool = True,
        window_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        if beta_max <= 0:
            raise ValueError("beta_max must be positive")
        if abs(beta_init) >= beta_max:
            raise ValueError("abs(beta_init) must be smaller than beta_max")
        if not 0.0 <= routing_floor < 1.0:
            raise ValueError("routing_floor must lie in [0, 1)")

        self.beta_max = float(beta_max)
        self.routing_floor = float(routing_floor)
        self.use_boundary = bool(use_boundary)
        self.detach_boundary = bool(detach_boundary)
        self.schedule_scale = 1.0
        self.local_adapter = OverlappingAmplitudeResidualAdapter(
            channels=channels,
            window_size=window_size,
            stride=stride,
            expansion=expansion,
            amplitude_scale=amplitude_scale,
            highpass_start=highpass_start,
            dropout=dropout,
            use_channel_mixing=use_channel_mixing,
            window_chunk_size=window_chunk_size,
        )
        initial_raw = math.atanh(float(beta_init) / float(beta_max))
        self.beta_raw = nn.Parameter(torch.tensor(initial_raw, dtype=torch.float32))
        self._last_stats: dict[str, float] = {}

    def set_schedule_scale(self, value: float) -> None:
        self.schedule_scale = float(min(max(value, 0.0), 1.0))

    def effective_beta(self) -> torch.Tensor:
        return self.schedule_scale * self.beta_max * torch.tanh(self.beta_raw)

    def forward(
        self,
        decoded: torch.Tensor,
        baseline_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        probability = torch.sigmoid(baseline_logits)
        boundary = _morphological_boundary(probability)
        if self.detach_boundary:
            boundary = boundary.detach()
        if self.use_boundary:
            route = self.routing_floor + (1.0 - self.routing_floor) * boundary
        else:
            route = torch.ones_like(boundary)

        local_residual = self.local_adapter(decoded)
        beta = self.effective_beta()
        refined = decoded + beta * route * local_residual

        with torch.no_grad():
            self._last_stats = {
                "dapr_baf/beta": float(beta.detach().item()),
                "dapr_baf/route_mean": float(route.mean().item()),
                "dapr_baf/route_min": float(route.min().item()),
                "dapr_baf/route_max": float(route.max().item()),
                "dapr_baf/boundary_mean": float(boundary.mean().item()),
                "dapr_baf/local_residual_abs_mean": float(
                    local_residual.abs().mean().item()
                ),
            }
        return refined, {
            "route": route,
            "boundary_evidence": boundary,
            "local_residual": local_residual,
            "beta": beta,
        }

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        values = dict(self._last_stats)
        values.update(self.local_adapter.diagnostics())
        return values


@register_model("dapr_baf_unet")
class DAPRBAFUNet(_PlainFourierUNetBase):
    """Direct amplitude--phase reconstruction with boundary amplitude refinement.

    The deepest encoder feature is replaced by a direct global amplitude--phase
    reconstruction. A lightweight overlapping local amplitude adapter then
    refines decoder features only around the coarse predicted boundary.
    """

    def __init__(
        self,
        *args,
        baf_enabled: bool = True,
        baf_window_size: int = 16,
        baf_stride: int = 8,
        baf_expansion: float = 1.0,
        baf_amplitude_scale: float = 0.20,
        baf_highpass_start: float = 0.20,
        baf_dropout: float = 0.05,
        baf_beta_max: float = 0.25,
        baf_beta_init: float = 0.02,
        baf_routing_floor: float = 0.05,
        baf_use_boundary: bool = True,
        baf_detach_boundary: bool = False,
        baf_use_channel_mixing: bool = True,
        baf_window_chunk_size: int = 256,
        baf_warmup_epochs: int = 0,
        **kwargs,
    ) -> None:
        kwargs["fourier_residual"] = False
        kwargs["fourier_zero_init_output"] = False
        kwargs.setdefault("fourier_use_amplitude", True)
        kwargs.setdefault("fourier_use_phase", True)
        super().__init__(*args, **kwargs)

        self.baf_enabled = bool(baf_enabled)
        self.baf_warmup_epochs = int(baf_warmup_epochs)
        if self.baf_enabled:
            decoder_channels = int(self.seg_head.in_channels)
            self.baf_refiner: BoundaryGuidedAmplitudeRefiner | None = (
                BoundaryGuidedAmplitudeRefiner(
                    channels=decoder_channels,
                    window_size=baf_window_size,
                    stride=baf_stride,
                    expansion=baf_expansion,
                    amplitude_scale=baf_amplitude_scale,
                    highpass_start=baf_highpass_start,
                    dropout=baf_dropout,
                    beta_max=baf_beta_max,
                    beta_init=baf_beta_init,
                    routing_floor=baf_routing_floor,
                    use_boundary=baf_use_boundary,
                    detach_boundary=baf_detach_boundary,
                    use_channel_mixing=baf_use_channel_mixing,
                    window_chunk_size=baf_window_chunk_size,
                )
            )
        else:
            self.baf_refiner = None
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        if not hasattr(self, "baf_refiner") or self.baf_refiner is None:
            return
        if self.baf_warmup_epochs <= 0:
            scale = 1.0
        else:
            scale = min(
                max((float(epoch) + 1.0) / float(self.baf_warmup_epochs), 0.0),
                1.0,
            )
        self.baf_refiner.set_schedule_scale(scale)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        decoded, _ = self._plain_forward_features(x, return_disagreement=False)
        baseline_logits = self.seg_head(decoded)
        if self.baf_refiner is None:
            return {
                "main": baseline_logits,
                "baseline_logits": baseline_logits,
            }
        refined, evidence = self.baf_refiner(decoded, baseline_logits)
        main_logits = self.seg_head(refined)
        return {
            "main": main_logits,
            "baseline_logits": baseline_logits,
            **evidence,
        }

    def diagnostic_metrics(self) -> dict[str, float]:
        values = super().diagnostic_metrics()
        if self.baf_refiner is not None:
            values.update(self.baf_refiner.diagnostics())
        return values

    def diagnostics(self) -> dict[str, float]:
        return self.diagnostic_metrics()


__all__ = [
    "OverlappingAmplitudeResidualAdapter",
    "BoundaryGuidedAmplitudeRefiner",
    "DAPRBAFUNet",
]
