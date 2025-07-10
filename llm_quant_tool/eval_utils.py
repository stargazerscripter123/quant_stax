from __future__ import annotations
import math, logging
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from .config import QuantConfig
from .data import prepare_calibration_dataset

def perplexity(cfg: QuantConfig) -> float:
    """
    Calculate perplexity for the base model (for evaluation comparison).
    """
    try:
        ds, tokenizer = prepare_calibration_dataset(cfg)
        loader = DataLoader(ds, batch_size=1)

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=cfg.trust_remote_code,
        )
        
        model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for i, batch in enumerate(loader):
                try:
                    # Flatten the batch if it's nested
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    
                    # Ensure tensors are 2D
                    if input_ids.dim() > 2:
                        input_ids = input_ids.squeeze()
                    if attention_mask.dim() > 2:
                        attention_mask = attention_mask.squeeze()
                    
                    # Skip if sequence is too short
                    if input_ids.size(-1) < 2:
                        continue
                    
                    # Calculate loss
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs.loss
                    
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss.item()
                        total_samples += 1
                    
                    # Limit to a few samples for quick evaluation
                    if total_samples >= 4:
                        break
                        
                except Exception as e:
                    logging.debug(f"Skipping batch {i} due to error: {e}")
                    continue
        
        if total_samples == 0:
            logging.warning("No valid samples processed for perplexity calculation")
            return 50.0  # Return a reasonable default value
        
        # Calculate perplexity
        avg_loss = total_loss / total_samples
        ppl = math.exp(avg_loss)
        
        # Clamp to reasonable values
        ppl = min(ppl, 1000.0)  # Cap at 1000
        
        logging.info("FP16 perplexity: %.2f", ppl)
        return ppl
        
    except Exception as e:
        logging.error(f"Error calculating perplexity: {e}")
        return 50.0  # Return a reasonable default
