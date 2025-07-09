"""
Activation‑aware Weight Quantisation (AWQ) wrapper using llm-compressor.
"""
from __future__ import annotations
import logging
from pathlib import Path
from ..config import QuantConfig

def quantise_awq(cfg: QuantConfig) -> Path:
    try:
        from llmcompressor.transformers import oneshot  # type: ignore
        from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install with: pip install llmcompressor") from e

    logging.info("Loading model and tokenizer …")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        device_map="auto",
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
    )

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
