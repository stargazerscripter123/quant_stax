"""
Model detection and configuration utilities.
Shared across all quantization backends for consistent model type detection.
"""
from __future__ import annotations
import json
import logging
import os
from typing import Dict, Any, Tuple, Optional


def detect_model_type(model_path: str) -> str:
    """
    Detect model type based on the model path or config.
    
    Args:
        model_path: Path to the model directory or HuggingFace model name
        
    Returns:
        Model type string (e.g., 'llama', 'qwen2', 'mixtral', etc.)
    """
    # First try to detect from config.json
    arch_type = detect_model_architecture(model_path)
    if arch_type:
        return arch_type
    
    # Fall back to path-based detection
    model_path_lower = model_path.lower()
    
    if "qwen3" in model_path_lower:
        return "qwen3"
    elif "qwen2.5" in model_path_lower or "qwen2_5" in model_path_lower:
        # Check if it's a vision-language model
        if "vl" in model_path_lower or "vision" in model_path_lower:
            return "qwen2.5_vl"
        return "qwen2.5"
    elif "qwen2" in model_path_lower:
        # Check if it's a vision-language model
        if "vl" in model_path_lower or "vision" in model_path_lower:
            return "qwen2_vl"
        return "qwen2"
    elif "llama" in model_path_lower:
        return "llama"
    elif "wizardlm" in model_path_lower or "mixtral" in model_path_lower:
        return "mixtral"
    elif "deepseek" in model_path_lower:
        return "deepseek"
    elif "decilm" in model_path_lower or "nemotron" in model_path_lower:
        return "decilm"
    else:
        logging.warning(f"Could not detect model type from path: {model_path}")
        logging.info("Defaulting to 'llama' configuration")
        return "llama"


def detect_model_architecture(model_path: str) -> str | None:
    """
    Detect model architecture by reading the config.json file.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Detected model type or None if detection fails
    """
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            arch = config.get("architectures", [])
            if arch:
                arch_name = arch[0].lower()
                logging.info(f"Detected architecture from config: {arch_name}")
                
                # Map architecture names to our model types
                if "qwen2vlforconditionalgeneration" in arch_name:
                    return "qwen2_vl"
                elif "qwen2_5vlforconditionalgeneration" in arch_name:
                    return "qwen2.5_vl"
                elif "qwen2forcausallm" in arch_name:
                    return "qwen2"
                elif "llamaforcausallm" in arch_name:
                    return "llama"
                elif "mixtralforcausallm" in arch_name:
                    return "mixtral"
                elif "deepseekcodeforcodingcausallm" in arch_name or "deepseek" in arch_name:
                    return "deepseek"
                elif "decilmforcausallm" in arch_name:
                    return "decilm"
        except Exception as e:
            logging.warning(f"Could not read config.json: {e}")
    
    return None


def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    Get model configuration for the specified model type.
    
    Args:
        model_type: Model type string
        
    Returns:
        Dictionary containing model configuration
    """
    # Common configurations for different model types
    configs = {
        "llama": {
            "model_class": "AutoModelForCausalLM",
            "tokenizer_class": "AutoTokenizer",
            "trust_remote_code": False,
            "use_fast": False,
            "torch_dtype": "auto"
        },
        "qwen2": {
            "model_class": "AutoModelForCausalLM",
            "tokenizer_class": "AutoTokenizer",
            "trust_remote_code": True,
            "use_fast": False,
            "torch_dtype": "auto"
        },
        "qwen2.5": {
            "model_class": "AutoModelForCausalLM",
            "tokenizer_class": "AutoTokenizer",
            "trust_remote_code": True,
            "use_fast": False,
            "torch_dtype": "auto"
        },
        "qwen3": {
            "model_class": "AutoModelForCausalLM",
            "tokenizer_class": "AutoTokenizer",
            "trust_remote_code": True,
            "use_fast": False,
            "torch_dtype": "auto"
        },
        "qwen2_vl": {
            "model_class": "Qwen2VLForConditionalGeneration",
            "tokenizer_class": "Qwen2VLProcessor",
            "trust_remote_code": True,
            "use_fast": False,
            "torch_dtype": "auto"
        },
        "qwen2.5_vl": {
            "model_class": "Qwen2VLForConditionalGeneration",
            "tokenizer_class": "Qwen2VLProcessor",
            "trust_remote_code": True,
            "use_fast": False,
            "torch_dtype": "auto"
        },
        "mixtral": {
            "model_class": "AutoModelForCausalLM",
            "tokenizer_class": "AutoTokenizer",
            "trust_remote_code": False,
            "use_fast": False,
            "torch_dtype": "auto"
        },
        "deepseek": {
            "model_class": "AutoModelForCausalLM",
            "tokenizer_class": "AutoTokenizer",
            "trust_remote_code": True,
            "use_fast": False,
            "torch_dtype": "auto"
        },
        "decilm": {
            "model_class": "AutoModelForCausalLM",
            "tokenizer_class": "AutoTokenizer",
            "trust_remote_code": True,
            "use_fast": False,
            "torch_dtype": "auto"
        }
    }
    
    # Return the specific config or fall back to llama
    return configs.get(model_type, configs["llama"])


def load_model_and_tokenizer(model_path: str, model_type: str) -> Tuple[Any, Any]:
    """
    Load model and tokenizer based on the detected model type.
    
    Args:
        model_path: Path to the model
        model_type: Detected model type
        
    Returns:
        Tuple of (model, tokenizer)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    config = get_model_config(model_type)
    logging.info(f"Loading model: {model_path}")
    logging.info(f"Model type: {model_type}")
    logging.info(f"Configuration: {config}")
    
    # Convert torch_dtype string to actual dtype
    torch_dtype = config["torch_dtype"]
    if torch_dtype == "auto":
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    elif isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype, torch.float32)
    
    # Import the appropriate classes
    if config["model_class"] == "AutoModelForCausalLM":
        model_class = AutoModelForCausalLM
    else:
        # For vision-language models, try to import them
        try:
            from transformers import Qwen2VLForConditionalGeneration
            if config["model_class"] == "Qwen2VLForConditionalGeneration":
                model_class = Qwen2VLForConditionalGeneration
            else:
                logging.warning(f"Unknown model class {config['model_class']}, falling back to AutoModelForCausalLM")
                model_class = AutoModelForCausalLM
        except ImportError:
            logging.warning("Vision-language models not available, falling back to AutoModelForCausalLM")
            model_class = AutoModelForCausalLM
    
    if config["tokenizer_class"] == "AutoTokenizer":
        tokenizer_class = AutoTokenizer
    else:
        # For vision processors
        try:
            from transformers import Qwen2VLProcessor
            if config["tokenizer_class"] == "Qwen2VLProcessor":
                tokenizer_class = Qwen2VLProcessor
            else:
                logging.warning(f"Unknown tokenizer class {config['tokenizer_class']}, falling back to AutoTokenizer")
                tokenizer_class = AutoTokenizer
        except ImportError:
            logging.warning("Vision processors not available, falling back to AutoTokenizer")
            tokenizer_class = AutoTokenizer
    
    # Load the model
    model = model_class.from_pretrained(
        model_path,
        trust_remote_code=config["trust_remote_code"],
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    
    # Load the tokenizer
    tokenizer = tokenizer_class.from_pretrained(
        model_path,
        trust_remote_code=config["trust_remote_code"],
        use_fast=config["use_fast"]
    )
    
    return model, tokenizer


def get_supported_model_types() -> list[str]:
    """
    Get list of supported model types.
    
    Returns:
        List of supported model type strings
    """
    return [
        "auto", "qwen2", "qwen2.5", "qwen3", "qwen2_vl", "qwen2.5_vl", 
        "llama", "mixtral", "deepseek", "decilm"
    ]
