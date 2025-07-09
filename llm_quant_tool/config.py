"""Global dataclass + YAML helpers (GPTQ params removed)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml, logging

@dataclass
class QuantConfig:
    # data / model
    model_name_or_path: str = "facebook/opt-125m"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_split: str = "validation"
    num_calibration_samples: int = 128
    block_size: int = 2048

    # method
    method: str = "llm_fp4"         # llm_fp4 | bnb_fp4 | fp8 | awq | gptq

    # LLM‑FP4 search
    search_rounds: int = 3
    search_intervals: list[float] = (0.01, 1.2, 100)

    # BnB‑FP4
    compute_dtype: str = "bfloat16"
    bnb_quant_type: str = "fp4"
    group_size: int = 128

    # SmoothQuant‑FP8
    alpha: float = 0.5

    # AWQ
    awq_bits: int = 4

    # GPTQ
    gptq_bits: int = 4
    gptq_group_size: int = 128
    gptq_desc_act: bool = False
    gptq_force_cpu: bool = False  # Force CPU mode for debugging

    # misc
    seed: int = 42
    out_dir: Path = Path("./quantized")
    trust_remote_code: bool = False

    # I/O
    @staticmethod
    def load(p: str | Path) -> "QuantConfig":
        with open(p, "r", encoding="utf-8") as f:
            return QuantConfig(**yaml.safe_load(f))

    def dump(self, p: str | Path):
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.__dict__, f)

def load_config(p: str | Path | None) -> QuantConfig:
    if p is None:
        logging.info("No --config given, defaulting to configs/llm_fp4.yaml")
        p = Path(__file__).parent.parent / "configs" / "llm_fp4.yaml"
    return QuantConfig.load(p)
