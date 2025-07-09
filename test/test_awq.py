from llm_quant_tool.quant import quantise
from llm_quant_tool.config import QuantConfig

def test_dispatch_awq():
    cfg = QuantConfig(method="awq", num_calibration_samples=1)
    try:
        quantise(cfg)
    except RuntimeError as e:
        # Should fail with llmcompressor import error if not installed
        assert "llmcompressor" in str(e)
