"""Dispatch to backend (GPTQ removed)."""
from __future__ import annotations
from pathlib import Path
from ..config import QuantConfig

def quantise(cfg: QuantConfig) -> Path:
    if cfg.method == "bnb_fp4":
        from .bnb_fp4 import quantise_bnb_fp4
        return quantise_bnb_fp4(cfg)
    if cfg.method == "fp8":
        from .fp8 import quantise_fp8
        return quantise_fp8(cfg)
    if cfg.method == "awq":
        from .awq import quantise_awq
        return quantise_awq(cfg)
    if cfg.method == "gptq":
        from .gptq import quantise_gptq
        return quantise_gptq(cfg)
    raise ValueError(f"Unknown method '{cfg.method}'")
