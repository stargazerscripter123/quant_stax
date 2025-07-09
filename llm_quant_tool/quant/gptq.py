"""
GPTQ quantization wrapper using GPTQModel library.
"""
from __future__ import annotations
import logging
from pathlib import Path
from ..config import QuantConfig

def quantise_gptq(cfg: QuantConfig) -> Path:
    try:
        from gptqmodel import GPTQModel, QuantizeConfig  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
        import torch  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install with: pip install gptqmodel") from e

    # Check GPU availability and memory
    if cfg.gptq_force_cpu:
        device = "cpu"
        logging.info("Forcing CPU mode for GPTQ (gptq_force_cpu=True)")
    elif torch.cuda.is_available():
        try:
            # Test GPU memory
            gpu_properties = torch.cuda.get_device_properties(0)
            available_memory = gpu_properties.total_memory / 1024**3  # GB
            
            if available_memory > 2.0:  # Need at least 2GB for small models
                device = "cuda"
                logging.info(f"Using GPU for GPTQ quantization (GPU: {gpu_properties.name}, {available_memory:.1f}GB)")
            else:
                device = "cpu"
                logging.warning(f"GPU memory too low ({available_memory:.1f}GB), using CPU")
        except Exception as e:
            device = "cpu"
            logging.warning(f"GPU check failed ({e}), falling back to CPU")
    else:
        device = "cpu"
        logging.warning("GPU not available, falling back to CPU (quantization will be slower)")

    logging.info("Loading model and tokenizer for GPTQ quantization …")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
    )
    
    # Set pad_token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create quantization configuration
    quantize_config = QuantizeConfig(
        bits=cfg.gptq_bits,  # Use FP4 mode (4 bits)
        group_size=cfg.gptq_group_size,
        desc_act=cfg.gptq_desc_act,
        static_groups=False,
        sym=True,
        true_sequential=True,
    )

    # Load model with explicit device placement
    model = GPTQModel.from_pretrained(
        cfg.model_name_or_path,
        quantize_config=quantize_config,
        trust_remote_code=cfg.trust_remote_code,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    # Ensure model is on the correct device
    if device == "cuda" and hasattr(model, 'cuda'):
        model = model.cuda()
        logging.info("Model moved to GPU for quantization")

    # Log GPU information for verification
    if device == "cuda":
        gpu_info = torch.cuda.get_device_properties(0)
        logging.info(f"GPU Device: {gpu_info.name}")
        logging.info(f"GPU Memory: {gpu_info.total_memory / 1024**3:.1f} GB")
        logging.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        logging.info(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")

    logging.info("Preparing calibration dataset for GPTQ …")
    
    # Simple dataset preparation for GPTQ
    from datasets import load_dataset  # type: ignore
    dataset = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split)
    
    # Prepare examples for GPTQModel - convert to proper tensor format
    # Based on successful GPTQModel examples, we need tensors on the correct device
    examples = []
    count = 0
    max_attempts = cfg.num_calibration_samples * 3  # Avoid infinite loop
    attempts = 0
    
    for item in dataset:
        if count >= cfg.num_calibration_samples or attempts >= max_attempts:
            break
        attempts += 1
        
        text = item["text"].strip()
        if len(text) > 10:  # Filter very short texts
            # Tokenize and convert to tensors on the correct device
            tokens = tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=1024,  # Model's context length to avoid warnings
                return_tensors="pt"  # Return PyTorch tensors
            )
            
            # Move tensors to the same device as the model and ensure proper format
            if "input_ids" in tokens and "attention_mask" in tokens:
                example = {
                    "input_ids": tokens["input_ids"].to(device),
                    "attention_mask": tokens["attention_mask"].to(device)
                }
                examples.append(example)
                count += 1
    
    logging.info(f"Prepared {len(examples)} calibration samples for GPTQ (tensors on {device})")

    logging.info("Running GPTQ quantization …")
    
    # GPU optimization settings
    if device == "cuda":
        # Clear GPU cache before quantization
        torch.cuda.empty_cache()
        logging.info("Cleared GPU cache before quantization")
        
        # Set GPU memory fraction if needed (optional)
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
    
    # Quantize the model (modifies model in-place)
    model.quantize(examples)
    
    if device == "cuda":
        logging.info("GPTQ quantization completed on GPU")
        # Clear cache after quantization
        torch.cuda.empty_cache()

    # Ensure output directory exists
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save quantized model (model is modified in-place)
    try:
        model.save_quantized(str(out_dir))
    except AttributeError:
        # Fallback method
        model.save_pretrained(str(out_dir))
    
    tokenizer.save_pretrained(out_dir)

    logging.info("GPTQ quantization completed. Model saved to %s", out_dir)
    return out_dir
