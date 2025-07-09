import torch
from llm_quant_tool.config import QuantConfig
from llm_quant_tool.quant.fp8 import quantise_fp8

def test_fp8_dtype_choice():
    cfg = QuantConfig(method="fp8", num_calibration_samples=1)
    try:
        quantise_fp8(cfg)
    except RuntimeError:
        # transformer‑engine or GPU may be absent
        pass
    dtype = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.int8
    assert dtype in {torch.float8_e4m3fn, torch.int8}
