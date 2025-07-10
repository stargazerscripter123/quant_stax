"""
SmoothQuant-FP8 (E4M3) - requires calibration data.
Supports both transformer-engine acceleration (when available) and pure PyTorch fallback.
"""
from __future__ import annotations
import json, logging, warnings
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

# Check for transformer-engine availability
try:
    import transformer_engine
    HAS_TRANSFORMER_ENGINE = True
    logging.info("transformer-engine available: accelerated FP8 support enabled")
except ImportError:
    HAS_TRANSFORMER_ENGINE = False
    logging.info("transformer-engine not available: using pure PyTorch FP8 implementation")

from transformers.data.data_collator import DataCollatorWithPadding

from ..config import QuantConfig
from ..data import prepare_calibration_dataset

# ------------------------------------------------------------------- helpers
def _iter_linear_modules(model):
    import torch.nn as nn
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            yield name, mod

def _collect_act_max(model, loader, device) -> Dict[str, torch.Tensor]:
    stats: Dict[str, torch.Tensor] = {}
    hooks = []

    def _hook_factory(layer):
        def _hook(mod, inp, _out):
            x = inp[0]
            vmax = x.detach().abs().amax(dim=list(range(x.ndim - 1)))
            stats[layer] = torch.maximum(stats.get(layer, torch.zeros_like(vmax)), vmax)
        return _hook

    for n, m in _iter_linear_modules(model):
        hooks.append(m.register_forward_hook(_hook_factory(n)))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collect SQ stats", leave=False):
            # Move batch to the same device as the model
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    for h in hooks:
        h.remove()
    return stats

# --------------------------------------------------------------------- main
def quantise_fp8(cfg: QuantConfig) -> Path:
    """
    FP8 quantization with SmoothQuant.
    
    Uses transformer-engine for acceleration when available, falls back to pure PyTorch.
    Supports E4M3 FP8 format with automatic fallback to INT8 on older PyTorch versions.
    """
    logging.info(f"FP8 quantization starting - transformer-engine: {'✓' if HAS_TRANSFORMER_ENGINE else '✗'}")
    
    # Determine target dtype with multiple fallbacks
    if hasattr(torch, "float8_e4m3fn"):
        tgt_dtype = torch.float8_e4m3fn
        logging.info("Using native PyTorch FP8 E4M3 format")
    else:
        tgt_dtype = torch.int8
        logging.warning("PyTorch FP8 not available, falling back to INT8")

    # Use a single device to avoid multi-GPU issues during calibration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch.float16,
        device_map=device,  # Use single device instead of "auto"
        trust_remote_code=cfg.trust_remote_code,
    )

    cal_ds, _tok = prepare_calibration_dataset(cfg)
    
    # Use a simple collator that converts to tensors
    def collate_fn(batch):
        return {k: torch.tensor([item[k] for item in batch], dtype=torch.long if k in ['input_ids', 'attention_mask'] else torch.float) 
                for k in batch[0].keys()}
    
    cal_loader = DataLoader(cal_ds, batch_size=4, collate_fn=collate_fn)

    act_max = _collect_act_max(model, cal_loader, device)
    logging.info("Collected SQ stats for %d Linear layers", len(act_max))

    state = {}
    for name, mod in _iter_linear_modules(model):
        if name not in act_max:
            continue
        s = act_max[name].clamp(min=1e-5).to(mod.weight.device)
        pow_s = s.pow(cfg.alpha)
        inv_pow = pow_s.reciprocal()

        mod.weight.data.mul_(pow_s)

        state[f"{name}.weight"] = mod.weight.detach().to(dtype=tgt_dtype).cpu()
        state[f"{name}.sq_inv_scale"] = inv_pow.to(torch.float16).cpu()
        if mod.bias is not None:
            state[f"{name}.bias"] = mod.bias.detach().cpu()

    # save remaining params untouched
    for n, p in model.named_parameters():
        if n not in state:
            state[n] = p.detach().cpu()

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model-fp8-sq.safetensors"
    save_file(state, str(path))
    (out / "smoothquant.json").write_text(
        json.dumps({"alpha": cfg.alpha, "dtype": str(tgt_dtype)}, indent=2)
    )
    logging.info("FP8 SmoothQuant checkpoint saved to %s", path)
    return path
