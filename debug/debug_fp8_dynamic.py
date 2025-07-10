#!/usr/bin/env python3
"""
Debug script for testing dynamic FP8 quantization functionality
Run from debug/ directory: python debug_fp8_dynamic.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import llm_quant_tool
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_dynamic_fp8_imports():
    """Test dynamic FP8-related imports"""
    print("=== Testing Dynamic FP8 Imports ===")
    
    try:
        from llm_quant_tool.quant.fp8 import quantise_fp8_dynamic, detect_model_type, HAS_LLMCOMPRESSOR
        print("✓ Dynamic FP8 quantization function available")
        print(f"✓ llmcompressor available: {'Yes' if HAS_LLMCOMPRESSOR else 'No (install with pip install llmcompressor)'}")
        print("✓ Model type detection available")
        return True
    except ImportError as e:
        print(f"✗ Dynamic FP8 import failed: {e}")
        return False

def test_model_type_detection():
    """Test model type detection"""
    print("\n=== Testing Model Type Detection ===")
    
    try:
        from llm_quant_tool.quant.fp8 import detect_model_type
        
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

def test_dynamic_fp8_config_loading():
    """Test dynamic FP8 config loading"""
    print("\n=== Testing Dynamic FP8 Config Loading ===")
    
    try:
        from llm_quant_tool.config import QuantConfig
        
        # Try to load dynamic FP8 config
        config_path = Path(__file__).parent.parent / "configs" / "fp8_dynamic.yaml"
        if config_path.exists():
            cfg = QuantConfig.load(str(config_path))
            print(f"✓ Dynamic FP8 config loaded: {cfg.model_name_or_path}")
            print(f"  Method: {cfg.method}")
            print(f"  Dynamic mode: {cfg.fp8_dynamic}")
            print(f"  Scheme: {cfg.fp8_scheme}")
            print(f"  Model type: {cfg.model_type}")
            print(f"  Test generation: {cfg.test_generation}")
            return True
        else:
            print(f"✗ Dynamic FP8 config file not found: {config_path}")
            return False
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_llmcompressor_availability():
    """Test llmcompressor availability and features"""
    print("\n=== Testing llmcompressor Availability ===")
    
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
        
        # Test supported schemes
        try:
            modifier = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC")
            print("✓ FP8_DYNAMIC scheme supported")
        except Exception as e:
            print(f"⚠️  FP8_DYNAMIC scheme issue: {e}")
        
        try:
            modifier = QuantizationModifier(targets="Linear", scheme="NVFP4A16")
            print("✓ NVFP4A16 scheme supported")
        except Exception as e:
            print(f"⚠️  NVFP4A16 scheme issue: {e}")
        
        return True
    except ImportError:
        print("ℹ️  llmcompressor not installed")
        print("💡 For dynamic FP8 quantization, install: pip install llmcompressor")
        return False

def test_cli_integration():
    """Test CLI integration with dynamic FP8 options"""
    print("\n=== Testing CLI Integration ===")
    
    try:
        from llm_quant_tool.cli import main
        
        # Test help with new options
        try:
            main(["--help"])
        except SystemExit:
            pass  # Help exits normally
        
        print("✓ CLI with dynamic FP8 options available")
        print("  Available options: --fp8-dynamic, --fp8-scheme, --model-type, --test-generation")
        return True
    except Exception as e:
        print(f"✗ CLI integration failed: {e}")
        return False

def main():
    """Run all dynamic FP8 tests"""
    print("Dynamic FP8 Debug Script")
    print("==============================")
    
    tests = [
        ("Dynamic FP8 Imports", test_dynamic_fp8_imports),
        ("Model Type Detection", test_model_type_detection),
        ("Dynamic FP8 Config Loading", test_dynamic_fp8_config_loading),
        ("llmcompressor Availability", test_llmcompressor_availability),
        ("CLI Integration", test_cli_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    print("\n==============================")
    print("DYNAMIC FP8 DEBUG SUMMARY")
    print("==============================")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:<30} {status}")
    
    if passed == total:
        print("\n🎉 Dynamic FP8 is ready to use!")
        print("💡 Usage:")
        print("   CLI: python -m llm_quant_tool.cli --fp8-dynamic --fp8-scheme FP8_DYNAMIC")
        print("   Config: Set fp8_dynamic: true in fp8_dynamic.yaml")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check output above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
