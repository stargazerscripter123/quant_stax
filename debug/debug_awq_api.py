#!/usr/bin/env python3
"""
AWQ API test script to verify AWQ quantization functionality
Run from debug/ directory: python debug_awq_api.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import llm_quant_tool
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_awq_imports():
    """Test AWQ-related imports"""
    print("=== Testing AWQ Imports ===")
    
    try:
        import llmcompressor
        print(f"✓ llmcompressor available: {llmcompressor.__version__}")
        return True
    except ImportError as e:
        print(f"✗ llmcompressor import failed: {e}")
        return False

def test_awq_dependencies():
    """Test AWQ dependencies"""
    print("\n=== Testing AWQ Dependencies ===")
    
    deps = ["torch", "transformers", "datasets", "numpy"]
    missing = []
    
    for dep in deps:
        try:
            module = __import__(dep)
            if hasattr(module, '__version__'):
                print(f"✓ {dep}: {module.__version__}")
            else:
                print(f"✓ {dep}: available")
        except ImportError:
            print(f"✗ {dep}: missing")
            missing.append(dep)
    
    # Check numpy version specifically for AWQ
    try:
        import numpy as np
        if np.__version__.startswith('2.'):
            print("⚠️  Warning: NumPy 2.x detected. AWQ may require NumPy <2.0")
            return False
        else:
            print(f"✓ NumPy version compatible: {np.__version__}")
    except ImportError:
        missing.append("numpy")
    
    return len(missing) == 0

def test_awq_basic_functionality():
    """Test basic AWQ functionality without full quantization"""
    print("\n=== Testing AWQ Basic Functionality ===")
    
    try:
        # Test the actual imports our AWQ implementation uses
        from llmcompressor.transformers import oneshot
        print("✓ llmcompressor.transformers.oneshot available")
        
        from llmcompressor.modifiers.quantization import QuantizationModifier
        print("✓ QuantizationModifier available (used by LLM Quant Tool)")
        
        # Test AWQ-specific module (alternative implementation)
        try:
            from llmcompressor.modifiers.awq import AWQModifier
            print("✓ AWQModifier available (alternative implementation)")
        except ImportError:
            print("ℹ️  AWQModifier not available (using QuantizationModifier)")
        
        print("ℹ️  Note: LLM Quant Tool uses QuantizationModifier with W4A16 scheme for AWQ")
        return True
        
    except ImportError as e:
        print(f"✗ AWQ functionality test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ AWQ basic test error: {e}")
        return False

def test_llm_quant_tool_awq():
    """Test AWQ through llm_quant_tool"""
    print("\n=== Testing LLM Quant Tool AWQ Integration ===")
    
    try:
        from llm_quant_tool.quant.awq import quantise_awq
        print("✓ AWQ quantization function available")
        return True
    except ImportError as e:
        print(f"✗ AWQ integration import failed: {e}")
        return False

def test_config_loading():
    """Test AWQ config loading"""
    print("\n=== Testing AWQ Config Loading ===")
    
    try:
        from llm_quant_tool.config import QuantConfig
        import yaml
        
        # Try to load AWQ config
        config_path = Path(__file__).parent.parent / "configs" / "awq.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            cfg = QuantConfig(**config_data)
            print(f"✓ AWQ config loaded: {cfg.model_name_or_path}")
            print(f"  Method: {cfg.method}")
            print(f"  Calibration samples: {cfg.num_calibration_samples}")
            return True
        else:
            print("✗ AWQ config file not found")
            return False
            
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def main():
    """Run AWQ API tests"""
    print("AWQ API Test Script\n" + "="*30)
    
    # Run tests
    imports_ok = test_awq_imports()
    deps_ok = test_awq_dependencies()
    basic_ok = test_awq_basic_functionality() if imports_ok else False
    integration_ok = test_llm_quant_tool_awq()
    config_ok = test_config_loading()
    
    # Summary
    print("\n" + "="*30)
    print("AWQ API TEST SUMMARY")
    print("="*30)
    print(f"Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Dependencies: {'✓ PASS' if deps_ok else '✗ FAIL'}")
    print(f"Basic functionality: {'✓ PASS' if basic_ok else '✗ FAIL'}")
    print(f"Integration: {'✓ PASS' if integration_ok else '✗ FAIL'}")
    print(f"Config loading: {'✓ PASS' if config_ok else '✗ FAIL'}")
    
    if imports_ok and deps_ok and basic_ok and integration_ok and config_ok:
        print("\n🎉 AWQ is ready to use!")
        return 0
    else:
        print("\n⚠️  Some AWQ issues found. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
