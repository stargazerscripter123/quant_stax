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
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    artefact = quantise(cfg)
    print(f"✔ Saved to {artefact}")

    if args.eval:
        print(f"Baseline PPL: {perplexity(cfg):.2f}")

if __name__ == "__main__":
    sys.exit(main())
