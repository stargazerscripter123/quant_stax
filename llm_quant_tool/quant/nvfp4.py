"""
NVFP4A16 quantization using llmcompressor with advanced model type detection.
Supports multiple model architectures with automatic detection.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Any

import torch

# Import shared model utilities
from .model_utils import detect_model_type, load_model_and_tokenizer

# Check for llmcompressor availability
try:
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.utils import dispatch_for_generation
    HAS_LLMCOMPRESSOR = True
    logging.info("llmcompressor available: NVFP4A16 quantization enabled")
except ImportError:
    HAS_LLMCOMPRESSOR = False
    oneshot = None
    QuantizationModifier = None
    dispatch_for_generation = None
    logging.info("llmcompressor not available: NVFP4A16 quantization disabled")

from ..config import QuantConfig

# ------------------------------------------------------------------- model detection
# ------------------------------------------------------------------- NVFP4 quantization
def quantise_nvfp4(cfg: QuantConfig) -> Path:
    """
    Apply NVFP4A16 quantization using llmcompressor.
    Integrated from nvfp4.py
    """
    if not HAS_LLMCOMPRESSOR or oneshot is None or QuantizationModifier is None:
        raise RuntimeError("llmcompressor not available. Install with: pip install llmcompressor")
    
    # Detect model type if set to auto
    model_type = getattr(cfg, 'model_type', 'auto')
    if model_type == "auto":
        model_type = detect_model_type(cfg.model_name_or_path)
        logging.info(f"Auto-detected model type: {model_type}")
    
    logging.info(f"Using NVFP4A16 quantization scheme")
    
    # Load model and tokenizer using shared utilities
    model, tokenizer = load_model_and_tokenizer(cfg.model_name_or_path, model_type)
    
    # Configure the quantization recipe for NVFP4A16
    # For Whisper models, we need to target all Linear layers including projections
    if model_type == "whisper":
        recipe = QuantizationModifier(
            targets="Linear", 
            scheme="NVFP4A16",
            ignore=[]  # Don't ignore any layers for Whisper
        )
        logging.info("Using Whisper-specific NVFP4A16 configuration (all Linear layers)")
    else:
        recipe = QuantizationModifier(
            targets="Linear", 
            scheme="NVFP4A16", 
            ignore=["lm_head"]
        )
        logging.info("Using standard NVFP4A16 configuration")
    
    # Set output directory
    output_dir = Path(cfg.out_dir) / f"model-nvfp4a16-{model_type}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Output directory: {output_dir}")
    
    # Apply quantization
    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        output_dir=str(output_dir)
    )
    
    # Test generation if requested
    test_generation = getattr(cfg, 'test_generation', False)
    if test_generation and dispatch_for_generation is not None:
        logging.info("Testing generation after quantization...")
        try:
            dispatch_for_generation(model)
            
            # Simple test prompt
            test_prompt = "Hello, how are you?"
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            logging.info(f"Input: {test_prompt}")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.info(f"Output: {response}")
        except Exception as e:
            logging.warning(f"Generation test failed: {e}")
    
    logging.info(f"NVFP4A16 quantization completed. Model saved to: {output_dir}")
    return output_dir / "model.safetensors" if (output_dir / "model.safetensors").exists() else output_dir