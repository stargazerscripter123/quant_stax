#!/usr/bin/env python3
from __future__ import annotations
import argparse, logging, sys
from .config import load_config
from .quant import quantise
from .evaluate import perplexity

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", help="path to YAML config")
    p.add_argument("--eval", action="store_true", help="baseline FP16 perplexity")
    
    # Dynamic FP8 options
    p.add_argument("--fp8-dynamic", action="store_true", 
                   help="use dynamic FP8 quantization (requires llmcompressor)")
    p.add_argument("--fp8-scheme", choices=["FP8_DYNAMIC", "NVFP4A16"], 
                   default="FP8_DYNAMIC", help="FP8 quantization scheme")
    p.add_argument("--model-type", choices=["auto", "qwen2", "qwen2.5", "qwen3", "llama", 
                                           "mixtral", "deepseek", "decilm"], 
                   default="auto", help="model architecture type")
    p.add_argument("--test-generation", action="store_true",
                   help="test generation after quantization")
    
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    
    # Override config with CLI arguments if provided
    if args.fp8_dynamic:
        cfg.fp8_dynamic = True
        cfg.fp8_scheme = args.fp8_scheme
        cfg.model_type = args.model_type
        cfg.test_generation = args.test_generation
        logging.info(f"Using dynamic FP8 quantization with scheme: {cfg.fp8_scheme}")
    
    artefact = quantise(cfg)
    print(f"✔ Saved to {artefact}")

    if args.eval:
        print(f"Baseline PPL: {perplexity(cfg):.2f}")

if __name__ == "__main__":
    sys.exit(main())
