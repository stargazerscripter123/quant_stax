# LLM Quant‑Tool 🪄

A unified PyTorch toolkit for Post-Training Quantization (PTQ) of Large Language Models with multiple backend support.

## Supported Quantization Methods

| `method`  | Backend                | Status | Calib data? | Description |
|-----------|------------------------|--------|-------------|-------------|
| `bnb_fp4` | **bitsandbytes FP4**   | ✅     | No          | 4-bit FP4 quantization using bitsandbytes |
| `awq`     | LLM Compressor AWQ     | ✅     | Yes         | Activation-aware Weight Quantization (INT4) |
| `gptq`    | **GPTQModel INT4**     | ✅     | Yes         | GPTQ 4-bit quantization with GPU acceleration |
| `fp8`     | SmoothQuant‑FP8        | ✅     | Yes         | FP8 E4M3 quantization with optional transformer-engine acceleration |

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

### BitsAndBytes FP4 (No calibration data needed)

```bash
python -m llm_quant_tool.cli -c configs/bnb_fp4.yaml
```

### AWQ INT4 (Requires calibration)

```bash
python -m llm_quant_tool.cli -c configs/awq.yaml
```

### GPTQ INT4 (Requires calibration, GPU accelerated)

```bash
python -m llm_quant_tool.cli -c configs/gptq.yaml
```

### SmoothQuant FP8 (Requires calibration)

```bash
python -m llm_quant_tool.cli -c configs/fp8.yaml
```

## Configuration

Each quantization method has its own configuration file in the `configs/` directory:

- `configs/bnb_fp4.yaml` - bitsandbytes FP4 configuration
- `configs/awq.yaml` - AWQ configuration
- `configs/gptq.yaml` - GPTQ configuration
- `configs/fp8.yaml` - SmoothQuant FP8 configuration

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
    ├── awq.py          # AWQ quantization
    ├── bnb_fp4.py      # BitsAndBytes FP4
    ├── gptq.py         # GPTQ quantization
    └── fp8.py          # SmoothQuant FP8

configs/                # Configuration files
├── awq.yaml
├── bnb_fp4.yaml
├── gptq.yaml
└── fp8.yaml

debug/                  # Debug and validation scripts
├── README.md
├── debug_run_all_tests.py
├── debug_gptq.py
├── debug_gptq_simple.py
├── debug_awq_api.py
├── debug_fp8.py
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
