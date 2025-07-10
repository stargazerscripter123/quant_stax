#!/usr/bin/env python3
from __future__ import annotations
import argparse, logging, sys, os

# Handle both direct execution and module import
if __name__ == "__main__" and __package__ is None:
    # Add parent directory to path for direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from llm_quant_tool.config import load_config
    from llm_quant_tool.quant import quantise
    from llm_quant_tool.eval_utils import perplexity
    from llm_quant_tool.quant.model_utils import get_supported_model_types
else:
    # Module import
    from .config import load_config
    from .quant import quantise
    from .eval_utils import perplexity
    from .quant.model_utils import get_supported_model_types

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", help="path to YAML config")
    p.add_argument("--eval", action="store_true", help="baseline FP16 perplexity")
    
    # Quantization method selection
    p.add_argument("--method", choices=["bnb_fp4", "awq", "gptq", "fp8", "nvfp4"],
                   help="quantization method (overrides config)")
    
    # Dynamic FP8 options
    p.add_argument("--fp8-dynamic", action="store_true", 
                   help="use dynamic FP8 quantization (requires llmcompressor)")
    p.add_argument("--fp8-scheme", choices=["FP8_DYNAMIC", "NVFP4A16"], 
                   default="FP8_DYNAMIC", help="FP8 quantization scheme")
    p.add_argument("--model-type", choices=get_supported_model_types(), 
                   default="auto", help="model architecture type")
    p.add_argument("--test-generation", action="store_true",
                   help="test generation after quantization")
    
    args = p.parse_args(argv)

    # Smart config selection based on CLI arguments
    config_file = args.config
    if config_file is None:
        if args.fp8_dynamic:
            config_file = "configs/fp8_dynamic.yaml"
            logging.info("Using dynamic FP8 config: configs/fp8_dynamic.yaml")
        elif args.method == "nvfp4":
            config_file = "configs/nvfp4.yaml"
            logging.info("Using NVFP4 config: configs/nvfp4.yaml")
        elif args.method == "fp8":
            config_file = "configs/fp8.yaml"
            logging.info("Using FP8 config: configs/fp8.yaml")
        elif args.method == "awq":
            config_file = "configs/awq.yaml"
            logging.info("Using AWQ config: configs/awq.yaml")
        elif args.method == "gptq":
            config_file = "configs/gptq.yaml"
            logging.info("Using GPTQ config: configs/gptq.yaml")
        elif args.method == "bnb_fp4":
            config_file = "configs/bnb_fp4.yaml"
            logging.info("Using BnB-FP4 config: configs/bnb_fp4.yaml")
        else:
            # Default fallback to match the default method in QuantConfig
            config_file = "configs/bnb_fp4.yaml"
            logging.info("No method specified, using default config: configs/bnb_fp4.yaml")
    
    cfg = load_config(config_file)
    
    # Override config with CLI arguments if provided
    if args.method:
        cfg.method = args.method
        logging.info(f"Using quantization method: {cfg.method}")
    
    if args.fp8_dynamic:
        cfg.fp8_dynamic = True
        cfg.fp8_scheme = args.fp8_scheme
        cfg.model_type = args.model_type
        cfg.test_generation = args.test_generation
        logging.info(f"Using dynamic FP8 quantization with scheme: {cfg.fp8_scheme}")
    
    if args.model_type != "auto":
        cfg.model_type = args.model_type
        logging.info(f"Using model type: {cfg.model_type}")
    
    if args.test_generation:
        cfg.test_generation = True
    
    artefact = quantise(cfg)
    print(f"✔ Saved to {artefact}")

    if args.eval:
        print(f"Baseline PPL: {perplexity(cfg):.2f}")

if __name__ == "__main__":
    sys.exit(main())
