"""Prepare calibration dataset."""
from __future__ import annotations
import logging
from datasets import load_dataset
from transformers import AutoTokenizer
from .config import QuantConfig

def prepare_calibration_dataset(cfg: QuantConfig):
    logging.info("Loading %s …", cfg.dataset_name)
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split)
    ds = ds.shuffle(seed=cfg.seed).select(range(cfg.num_calibration_samples))

    tok = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    
    # Set pad_token if it doesn't exist
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _tok(batch):
        return tok(
            batch["text"],
            max_length=cfg.block_size,
            truncation=True,
            padding="max_length",
            return_tensors=None,  # Return lists, let DataLoader handle tensor conversion
        )

    ds = ds.map(_tok, batched=True, remove_columns=ds.column_names)
    logging.info("Prepared %d calibration samples", len(ds))
    return ds, tok
