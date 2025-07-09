# LLM Quant‑Tool 🪄

A unified PyTorch toolkit for Post-Training Quantization (PTQ) of Large Language Models with multiple backend support.

## Supported Quantization Methods

| `method`  | Backend                | Calib data? | Description |
|-----------|------------------------|-------------|-------------|
| `bnb_fp4` | **bitsandbytes FP4**   | No          | 4-bit FP4 quantization using bitsandbytes |
| `fp8`     | SmoothQuant‑FP8        | Yes         | FP8 E4M3 quantization with SmoothQuant |
| `awq`     | LLM Compressor AWQ     | Yes         | Activation-aware Weight Quantization (INT4) |
| `gptq`    | **GPTQModel FP4**      | Yes         | GPTQ 4-bit quantization with GPU acceleration |

## Installation

### Quick Start
```bash
# Install base dependencies
pip install -e .

# Install specific quantization backends
pip install -e .[bnb_fp4]     # bitsandbytes FP4
pip install -e .[fp8]         # SmoothQuant FP8 (Linux only)
pip install -e .[awq]         # LLM Compressor AWQ
pip install -e .[gptq]        # GPTQModel GPTQ

# Install all backends
pip install -e .[bnb_fp4,fp8,awq,gptq]
```

### Using Conda Environment
```bash
# Create and activate environment
conda env create -f environment.yml
conda activate llm-quant-tool

# The environment.yml includes all optional dependencies
```

## Usage Examples

### BitsAndBytes FP4 (No calibration data needed)
```bash
python -m llm_quant_tool.cli -c configs/bnb_fp4.yaml
```

### SmoothQuant FP8 (Requires calibration)
```bash
python -m llm_quant_tool.cli -c configs/fp8.yaml
```

### AWQ INT4 (Requires calibration)
```bash
python -m llm_quant_tool.cli -c configs/awq.yaml
```

### GPTQ INT4 (Requires calibration, GPU accelerated)
```bash
python -m llm_quant_tool.cli -c configs/gptq.yaml
```

## Configuration

Each quantization method has its own configuration file in the `configs/` directory:
- `configs/bnb_fp4.yaml` - bitsandbytes FP4 configuration
- `configs/fp8.yaml` - SmoothQuant FP8 configuration  
- `configs/awq.yaml` - AWQ configuration
- `configs/gptq.yaml` - GPTQ configuration

## Requirements

- Python 3.9+
- PyTorch 2.2.0+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ GPU memory (recommended for larger models)

## Project Structure

```
llm_quant_tool/
├── cli.py              # Command-line interface
├── config.py           # Configuration handling
├── data.py             # Dataset preparation
├── evaluate.py         # Model evaluation utilities
└── quant/              # Quantization backends
    ├── __init__.py     # Quantization dispatcher
    ├── awq.py          # AWQ quantization
    ├── bnb_fp4.py      # BitsAndBytes FP4
    ├── fp8.py          # SmoothQuant FP8
    └── gptq.py         # GPTQ quantization

configs/                # Configuration files
├── awq.yaml
├── bnb_fp4.yaml
├── fp8.yaml
└── gptq.yaml
```

### Running Tests
```bash
pip install -e .[dev]
pytest test/
```
