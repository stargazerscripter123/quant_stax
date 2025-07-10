"""
Activation‑aware Weight Quantisation (AWQ) wrapper using llm-compressor.
"""
from __future__ import annotations
import logging
from pathlib import Path
from ..config import QuantConfig

# Import shared model utilities
from .model_utils import detect_model_type, load_model_and_tokenizer

def quantise_awq(cfg: QuantConfig) -> Path:
    try:
        from llmcompressor.transformers import oneshot  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install with: pip install llmcompressor") from e

    logging.info("Loading model and tokenizer …")
    
    # Detect model type if set to auto
    model_type = getattr(cfg, 'model_type', 'auto')
    if model_type == "auto":
        model_type = detect_model_type(cfg.model_name_or_path)
        logging.info(f"Auto-detected model type: {model_type}")
    
    # Load model and tokenizer using shared utilities
    model, tokenizer = load_model_and_tokenizer(cfg.model_name_or_path, model_type)

    logging.info("Running AWQ quantization with llm-compressor …")
    
    # Create AWQ recipe for llm-compressor
    recipe = f"""
quant_stage:
  quant_modifiers:
    QuantizationModifier:
      targets: ["Linear"]
      scheme: "W{cfg.awq_bits}A16"
      ignore: ["lm_head"]
"""

    # Ensure output directory exists
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run oneshot quantization
    oneshot(
        model=model,
        dataset=cfg.dataset_name if cfg.dataset_name != "wikitext" else "open_platypus",
        recipe=recipe,
        max_seq_length=cfg.block_size,
        num_calibration_samples=cfg.num_calibration_samples,
        output_dir=str(out_dir),
        tokenizer=tokenizer,
        save_compressed=True,
    )

    logging.info("AWQ quantization completed. Model saved to %s", out_dir)
    return out_dir
