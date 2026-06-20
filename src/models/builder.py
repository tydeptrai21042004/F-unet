from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

from .registry import create_model


def _to_dict(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return deepcopy(config)
    return deepcopy(dict(config))


ALIASES = {
    "fourier": "plain_fourier_unet",
    "fourier_unet": "plain_fourier_unet",
    "plain_fourier": "plain_fourier_unet",
    "apdr": "apdr_fourier_unet",
    "apdr_unet": "apdr_fourier_unet",
}


def build_model(
    name: str,
    config: Optional[Mapping[str, Any]] = None,
    **overrides: Any,
):
    cfg = _to_dict(config)
    cfg.pop("name", None)
    cfg.update(overrides)
    cfg.pop("name", None)
    model_name = ALIASES.get(name.lower(), name.lower())
    return create_model(model_name, **cfg)


__all__ = ["build_model"]
