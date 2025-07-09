from llm_quant_tool.config import load_config

def test_default_cfg():
    cfg = load_config(None)
    assert cfg.method == "fp4"
    assert cfg.model_name_or_path
