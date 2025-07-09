import pytest
from llm_quant_tool.config import QuantConfig

def test_dispatch_gptq():
    cfg = QuantConfig(method="gptq", num_calibration_samples=1)
    try:
        from llm_quant_tool.quant.gptq import quantise_gptq
        # Test that function exists and is callable
        assert callable(quantise_gptq)
    except ImportError:
        pytest.skip("GPTQModel not installed")
