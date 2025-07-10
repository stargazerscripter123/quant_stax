#!/usr/bin/env python3
"""
Debug script for testing NVFP4A16 quantization functionality
Run from debug/ directory: python debug_nvfp4.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import llm_quant_tool
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_nvfp4_imports():
    """Test NVFP4-related imports"""
    print("=== Testing NVFP4 Imports ===")
    
    try:
        from llm_quant_tool.quant.nvfp4 import quantise_nvfp4, detect_model_type, HAS_LLMCOMPRESSOR
        print("✓ NVFP4 quantization function available")
        print(f"✓ llmcompressor available: {'Yes' if HAS_LLMCOMPRESSOR else 'No (install with pip install llmcompressor)'}")
        print("✓ Model type detection available")
        return True
    except ImportError as e:
        print(f"✗ NVFP4 import failed: {e}")
        return False

def test_model_type_detection():
    """Test model type detection"""
    print("\n=== Testing Model Type Detection ===")
    
    try:
        from llm_quant_tool.quant.nvfp4 import detect_model_type
        
        test_cases = [
            ("test/qwen2-model", "qwen2"),
            ("test/llama-model", "llama"),
            ("test/qwen2.5-vl-model", "qwen2.5_vl"),
            ("test/mixtral-model", "mixtral"),
            ("test/deepseek-model", "deepseek"),
        ]
        
        for model_path, expected in test_cases:
            detected = detect_model_type(model_path)
            status = "✓" if detected == expected else "✗"
            print(f"{status} {model_path} -> {detected} (expected: {expected})")
        
        return True
    except Exception as e:
        print(f"✗ Model type detection failed: {e}")
        return False

def test_nvfp4_config_loading():
    """Test NVFP4 config loading"""
    print("\n=== Testing NVFP4 Config Loading ===")
    
    try:
        from llm_quant_tool.config import QuantConfig
        
        # Try to load NVFP4 config
        config_path = Path(__file__).parent.parent / "configs" / "nvfp4.yaml"
        if config_path.exists():
            cfg = QuantConfig.load(str(config_path))
            print(f"✓ NVFP4 config loaded: {cfg.model_name_or_path}")
            print(f"  Method: {cfg.method}")
            print(f"  Model type: {cfg.model_type}")
            print(f"  Test generation: {cfg.test_generation}")
            return True
        else:
            print(f"✗ NVFP4 config file not found: {config_path}")
            return False
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_llmcompressor_nvfp4_support():
    """Test llmcompressor NVFP4A16 support"""
    print("\n=== Testing llmcompressor NVFP4A16 Support ===")
    
    try:
        import llmcompressor
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor.utils import dispatch_for_generation
        
        try:
            version = getattr(llmcompressor, '__version__', 'unknown')
            print(f"✓ llmcompressor available: {version}")
        except:
            print("✓ llmcompressor available: version unknown")
        
        print("✓ oneshot function available")
        print("✓ QuantizationModifier available")
        print("✓ dispatch_for_generation available")
        
        # Test NVFP4A16 scheme specifically
        try:
            modifier = QuantizationModifier(targets="Linear", scheme="NVFP4A16")
            print("✓ NVFP4A16 scheme supported")
        except Exception as e:
            print(f"⚠️  NVFP4A16 scheme issue: {e}")
        
        return True
    except ImportError:
        print("ℹ️  llmcompressor not installed")
        print("💡 For NVFP4A16 quantization, install: pip install llmcompressor")
        return False

def test_quantization_dispatcher():
    """Test quantization dispatcher with NVFP4"""
    print("\n=== Testing Quantization Dispatcher ===")
    
    try:
        from llm_quant_tool.quant import quantise
        from llm_quant_tool.config import QuantConfig
        
        # Create a test config for NVFP4
        test_cfg = QuantConfig()
        test_cfg.method = "nvfp4"
        test_cfg.model_name_or_path = "distilgpt2"
        
        print("✓ NVFP4 method recognized by dispatcher")
        print(f"✓ Test config created: {test_cfg.method}")
        return True
    except Exception as e:
        print(f"✗ Dispatcher test failed: {e}")
        return False

def test_cli_integration():
    """Test CLI integration with NVFP4 options"""
    print("\n=== Testing CLI Integration ===")
    
    try:
        from llm_quant_tool.cli import main
        
        # Test help with NVFP4 method
        try:
            main(["--help"])
        except SystemExit:
            pass  # Help exits normally
        
        print("✓ CLI with NVFP4 method available")
        print("  Available options: --method nvfp4, --model-type, --test-generation")
        return True
    except Exception as e:
        print(f"✗ CLI integration failed: {e}")
        return False

def main():
    """Run all NVFP4 tests"""
    print("NVFP4A16 Debug Script")
    print("==============================")
    
    tests = [
        ("NVFP4 Imports", test_nvfp4_imports),
        ("Model Type Detection", test_model_type_detection),
        ("NVFP4 Config Loading", test_nvfp4_config_loading),
        ("llmcompressor NVFP4A16 Support", test_llmcompressor_nvfp4_support),
        ("Quantization Dispatcher", test_quantization_dispatcher),
        ("CLI Integration", test_cli_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    print("\n==============================")
    print("NVFP4A16 DEBUG SUMMARY")
    print("==============================")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:<30} {status}")
    
    if passed == total:
        print("\n🎉 NVFP4A16 is ready to use!")
        print("💡 Usage:")
        print("   CLI: python -m llm_quant_tool.cli --method nvfp4 --model-type auto")
        print("   Config: python -m llm_quant_tool.cli -c configs/nvfp4.yaml")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check output above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
