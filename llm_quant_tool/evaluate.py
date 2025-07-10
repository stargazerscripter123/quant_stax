from __future__ import annotations
import math, logging
import torch
from torch.utils.data import DataLoader
try:
    from datasets import load_metric
except ImportError:
    # load_metric was deprecated, use evaluate instead
    from evaluate import load as load_metric
from transformers import AutoModelForCausalLM
from .config import QuantConfig
from .data import prepare_calibration_dataset

def perplexity(cfg: QuantConfig) -> float:
    ds, _ = prepare_calibration_dataset(cfg)
    loader = DataLoader(ds, batch_size=4)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=cfg.trust_remote_code,
    )
    metric = load_metric("perplexity")
    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            metric.add_batch(
                predictions=model(**batch).logits,
                references=batch["input_ids"],
            )
    ppl = math.exp(metric.compute()["mean_cross_entropy"])
    logging.info("FP16 perplexity: %.2f", ppl)
    return ppl
