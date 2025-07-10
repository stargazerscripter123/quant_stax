# Debug Scripts

---

### 🔍 `debug_validate_install.py`

**Purpose**: Comprehensive validation of the entire LLM Quant Tool installation
**Usage**: `python debug_validate_install.py`
**What it tests**:

- Core module imports (CLI, config, data, quantization dispatcher)
- All quantization backend availability
- External dependencies (torch, transformers, etc.)
- Package versions and compatibility
- CLI functionality

**When to use**: After installation or when troubleshooting general issues

---

### ⚡ `debug_gptq_simple.py`

**Purpose**: Quick GPTQ backend testing without actual quantization
**Usage**: `python debug_gptq_simple.py`
**What it tests**:

- GPTQModel imports and basic setup
- Device availability (CUDA vs CPU)
- Simple model initialization
- LLM Quant Tool GPTQ integration

**When to use**: Fast GPTQ troubleshooting or before running full quantization

---

### 🚀 `debug_gptq.py`

**Purpose**: Complete GPTQ workflow testing with actual quantization
**Usage**: `python debug_gptq.py`
**What it tests**:

- Memory usage and requirements
- CLI integration
- Full GPTQ quantization workflow (optional)
- Output file generation

**When to use**: Deep GPTQ debugging or validating full workflow
**Note**: Includes interactive prompt for full quantization test

---

### 📊 `debug_awq_api.py`

**Purpose**: AWQ backend testing and API validation
**Usage**: `python debug_awq_api.py`
**What it tests**:

- llmcompressor imports and version
- NumPy compatibility (AWQ requires numpy<2.0)
- AWQ basic functionality
- LLM Quant Tool AWQ integration
- AWQ config file loading

**When to use**: AWQ-specific troubleshooting or numpy compatibility issues

---

### 🔥 `debug_fp8.py`

**Purpose**: FP8 quantization testing and transformer-engine validation
**Usage**: `python debug_fp8.py`
**What it tests**:

- FP8 quantization imports and functionality
- PyTorch FP8 E4M3 format support
- transformer-engine availability (optional acceleration)
- FP8 config file loading
- Pure PyTorch fallback capability

**When to use**: FP8-specific troubleshooting or transformer-engine compatibility issues

---

### 🎯 `debug_run_all_tests.py`

**Purpose**: Comprehensive test runner that executes all debug scripts
**Usage**: `python debug_run_all_tests.py`
**What it tests**:

- Runs all validation scripts in sequence
- Provides comprehensive test suite summary
- Timeout protection for long-running tests
- Overall system health check

**When to use**: Complete system validation or CI/CD pipeline testing

