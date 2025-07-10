# LLM Quant‑Tool 🪄

A unified PyTorch toolkit for Post-Training Quantization (PTQ) of Large Language Models with multiple backend support.

## Supported Quantization Methods

| `method`  | Backend                | Status | Description |
|-----------|------------------------|--------|-------------|
| `bnb_fp4` | **bitsandbytes FP4**   | ✅     | 4-bit FP4 quantization using bitsandbytes |
| `awq`     | LLM Compressor AWQ     | ✅     | Activation-aware Weight Quantization (INT4) |
| `gptq`    | **GPTQModel INT4**     | ✅     | GPTQ 4-bit quantization with GPU acceleration |
| `fp8`     | SmoothQuant‑FP8        | ✅     | FP8 E4M3 quantization with optional transformer-engine acceleration |
| `fp8`     | **NVFP8**              | ✅     | NVFP8 quantization with auto model detection |
| `nvfp4`   | **NVFP4A16**           | ✅     | NVIDIA FP4A16 quantization with advanced model detection |

## Installation

### Quick Start

```bash
# Install base dependencies
pip install -e .

# Install specific quantization backends
pip install -e .[bnb_fp4]     # bitsandbytes FP4
pip install -e .[awq]         # LLM Compressor AWQ
pip install -e .[gptq]        # GPTQModel GPTQ
pip install -e .[fp8]         # SmoothQuant FP8 (with transformer-engine on Linux)

# Install all backends
pip install -e .[all]         # All backends including FP8

# Development installation
pip install -e .[dev]         # Testing and linting tools
```

### Using Conda Environment (Recommended)

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate llm-quant-tool

# The environment.yml includes all working dependencies with proper constraints
```

## Usage Examples

### test the quantization on distillgpt2

```bash
bash scripts/quantize_example.sh 
```

### BitsAndBytes FP4

```bash
python -m llm_quant_tool.cli -c configs/bnb_fp4.yaml
```

### AWQ INT4

```bash
python -m llm_quant_tool.cli -c configs/awq.yaml
```

### GPTQ INT4

```bash
python -m llm_quant_tool.cli -c configs/gptq.yaml
```

### FP8

```bash
python -m llm_quant_tool.cli -c configs/fp8.yaml
python -m llm_quant_tool.cli -c configs/fp8_dynamic.yaml

# With generation testing
python -m llm_quant_tool.cli --fp8-dynamic --test-generation
```

### NVFP4A16

```bash
python -m llm_quant_tool.cli -c configs/nvfp4.yaml

# With generation testing
python -m llm_quant_tool.cli --method nvfp4 --test-generation
```

## Configuration

Each quantization method has its own configuration file in the `configs/` directory:

- `configs/bnb_fp4.yaml` - bitsandbytes FP4 configuration
- `configs/awq.yaml` - AWQ configuration
- `configs/gptq.yaml` - GPTQ configuration
- `configs/fp8.yaml` - SmoothQuant FP8 configuration
- `configs/fp8_dynamic.yaml` - Dynamic FP8 configuration
- `configs/nvfp4.yaml` - NVFP4A16 configuration

## Model Type Detection

The toolkit includes automatic model type detection that works across all quantization backends:

### Supported Model Types

- **`auto`** - Automatic detection (default)
- **`llama`** - LLaMA family models (Llama 2, Llama 3, etc.)
- **`qwen2`** - Qwen2 models  
- **`qwen2.5`** - Qwen2.5 models
- **`qwen3`** - Qwen3 models
- **`qwen2_vl`** - Qwen2-VL vision-language models
- **`qwen2.5_vl`** - Qwen2.5-VL vision-language models
- **`mixtral`** - Mixtral models
- **`deepseek`** - DeepSeek models
- **`decilm`** - DeciLM/Nemotron models

### Detection Methods

1. **Architecture detection**: Reads `config.json` to identify model architecture
2. **Path-based detection**: Analyzes model name/path for keywords
3. **Fallback**: Defaults to LLaMA configuration for unknown models

### Usage

```bash
# Automatic detection (recommended)
python -m llm_quant_tool.cli --method bnb_fp4 --model-type auto

# Explicit model type
python -m llm_quant_tool.cli --method awq --model-type qwen2.5

# Vision-language models
python -m llm_quant_tool.cli --method nvfp4 --model-type qwen2_vl
```

## Requirements

- Python 3.9+
- PyTorch 2.2.0+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ GPU memory (recommended for larger models)

## Project Structure

```text
llm_quant_tool/
├── cli.py              # Command-line interface
├── config.py           # Configuration handling
├── data.py             # Dataset preparation
├── evaluate.py         # Model evaluation utilities
└── quant/              # Quantization backends
    ├── __init__.py     # Quantization dispatcher
    ├── model_utils.py  # Shared model detection utilities
    ├── awq.py          # AWQ quantization
    ├── bnb_fp4.py      # BitsAndBytes FP4
    ├── gptq.py         # GPTQ quantization
    ├── nvfp4.py        # NVFP4A16 quantization
    └── fp8.py          # SmoothQuant FP8 & Dynamic FP8

configs/                # Configuration files
├── awq.yaml
├── bnb_fp4.yaml
├── gptq.yaml
├── fp8.yaml
├── fp8_dynamic.yaml
├── nvfp4.yaml
└── llm_fp4.yaml

debug/                  # Debug and validation scripts
├── README.md
├── debug_run_all_tests.py
├── debug_gptq.py
├── debug_gptq_simple.py
├── debug_awq_api.py
├── debug_fp8.py
├── debug_nvfp4.py
└── debug_validate_install.py
└── debug_validate_install.py
```

## Development and Testing

### Running Tests

```bash
pip install -e .[dev]
pytest test/

# Run debug scripts for validation
cd debug/
python run_all_tests.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
