from __future__ import annotations
import logging
from pathlib import Path
from ..config import QuantConfig

# Import shared model utilities
from .model_utils import detect_model_type, get_model_config

def quantise_bnb_fp4(cfg: QuantConfig) -> Path:
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        import torch, bitsandbytes  # noqa: F401
    except ImportError as e:
        raise RuntimeError("Install bitsandbytes (`pip install .[bnb_fp4]`)") from e

    # Detect model type if set to auto
    model_type = getattr(cfg, 'model_type', 'auto')
    if model_type == "auto":
        model_type = detect_model_type(cfg.model_name_or_path)
        logging.info(f"Auto-detected model type: {model_type}")

    # Get model configuration
    model_config = get_model_config(model_type)
    logging.info(f"Using model type: {model_type}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, cfg.compute_dtype),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_group_size=cfg.group_size,
        bnb_4bit_quant_storage=cfg.compute_dtype,  # Use the same dtype as compute_dtype
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=model_config["trust_remote_code"],
        torch_dtype=model_config["torch_dtype"]
    )

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    logging.info("Bits‑and‑Bytes FP4 checkpoint saved to %s", out)
    return out
