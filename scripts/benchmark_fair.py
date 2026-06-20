#!/usr/bin/env python3
"""Run the canonical fair benchmark."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = ['unet', 'unetpp', 'attention_unet', 'pranet', 'acsnet', 'hardnet_mseg', 'cfanet', 'polyp_pvt', 'caranet', 'hsnet', 'resunetpp', 'plain_fourier_unet', 'apdr_fourier_unet']

def main() -> None:
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "benchmark_all.py"),
        "--config-dir", "configs/fair",
        "--models", ",".join(MODELS),
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
