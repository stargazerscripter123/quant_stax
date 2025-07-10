#!/usr/bin/env python3
"""
Validation script to test all LLM Quant Tool components and backends
Run from debug/ directory: python debug_validate_install.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import llm_quant_tool
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_core_imports():
    """Test core module imports"""
    print("=== Testing Core Imports ===")
    
    try:
        from llm_quant_tool.cli import main
        print("✓ CLI import successful")
    except ImportError as e:
        print(f"✗ CLI import failed: {e}")
        return False
    
    try:
        from llm_quant_tool.config import QuantConfig
        print("✓ Config import successful")
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        from llm_quant_tool.data import prepare_calibration_dataset
        print("✓ Data import successful")
    except ImportError as e:
        print(f"✗ Data import failed: {e}")
        return False
    
    try:
        from llm_quant_tool.quant import quantise
        print("✓ Quantization dispatcher successful")
    except ImportError as e:
        print(f"✗ Quantization dispatcher failed: {e}")
        return False
    
    return True

def test_backend_imports():
    """Test quantization backend imports"""
    print("\n=== Testing Quantization Backends ===")
    
    backends = {
        "bnb_fp4": "llm_quant_tool.quant.bnb_fp4",
        "awq": "llm_quant_tool.quant.awq",
        "gptq": "llm_quant_tool.quant.gptq",
        "fp8": "llm_quant_tool.quant.fp8"
    }
    
    working_backends = []
    for name, module in backends.items():
        try:
            __import__(module)
            print(f"✓ {name} backend import successful")
            working_backends.append(name)
        except ImportError as e:
            print(f"✗ {name} backend import failed: {e}")
    
    return working_backends

def test_external_dependencies():
    """Test external library availability"""
    print("\n=== Testing External Dependencies ===")
    
    external_deps = {
        "torch": "torch",
        "transformers": "transformers",
        "datasets": "datasets",
        "numpy": "numpy",
        "bitsandbytes": "bitsandbytes",
        "gptqmodel": "gptqmodel", 
        "llmcompressor": "llmcompressor",
        "evaluate": "evaluate"
    }
    
    available_deps = []
    for name, module in external_deps.items():
        try:
            __import__(module)
            print(f"✓ {name} available")
            available_deps.append(name)
        except ImportError:
            print(f"✗ {name} not available")
    
    return available_deps

def test_package_versions():
    """Test and display key package versions"""
    print("\n=== Package Versions ===")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
    except ImportError:
        print("PyTorch: Not available")
    
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers: Not available")
    
    try:
        import numpy
        print(f"NumPy: {numpy.__version__}")
        if numpy.__version__.startswith('2.'):
            print("⚠️  Warning: NumPy 2.x may cause llmcompressor issues")
    except ImportError:
        print("NumPy: Not available")
    
    try:
        import gptqmodel
        print(f"GPTQModel: {gptqmodel.__version__}")
    except ImportError:
        print("GPTQModel: Not available")

def test_cli_functionality():
    """Test CLI help command"""
    print("\n=== Testing CLI Functionality ===")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "llm_quant_tool.cli", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            print("✓ CLI help command works")
            return True
        else:
            print(f"✗ CLI help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("LLM Quant Tool Validation\n" + "="*40)
    
    # Test core functionality
    core_ok = test_core_imports()
    working_backends = test_backend_imports()
    available_deps = test_external_dependencies()
    test_package_versions()
    cli_ok = test_cli_functionality()
    
    # Summary
    print("\n" + "="*40)
    print("VALIDATION SUMMARY")
    print("="*40)
    print(f"Core imports: {'✓ PASS' if core_ok else '✗ FAIL'}")
    print(f"Working backends: {len(working_backends)}/4 ({', '.join(working_backends)})")
    print(f"Available dependencies: {len(available_deps)}")
    print(f"CLI functionality: {'✓ PASS' if cli_ok else '✗ FAIL'}")
    
    if core_ok and working_backends and cli_ok:
        print("\n🎉 LLM Quant Tool is ready to use!")
        return 0
    else:
        print("\n⚠️  Some issues found. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
