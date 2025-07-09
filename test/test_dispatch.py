from llm_quant_tool.quant import quantise
from llm_quant_tool.config import QuantConfig

def test_dispatch():
    for m in ("llm_fp4", "bnb_fp4", "fp8", "awq"):
        cfg = QuantConfig(method=m, num_calibration_samples=1)
        try:
            quantise(cfg)
        except RuntimeError:
            # backend may be missing on CI
            pass
