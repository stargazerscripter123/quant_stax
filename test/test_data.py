from llm_quant_tool.config import QuantConfig
from llm_quant_tool.data import prepare_calibration_dataset

def test_prepare_dataset():
    cfg = QuantConfig(num_calibration_samples=2)
    ds, _ = prepare_calibration_dataset(cfg)
    assert len(ds) == 2
    assert "input_ids" in ds[0]
