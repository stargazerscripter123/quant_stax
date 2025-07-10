#!/usr/bin/env python3
"""
FP8 debug script to test SmoothQuant FP8 quantization functionality
Run from debug/ directory: python debug_fp8.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import llm_quant_tool
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_fp8_imports():
    """Test FP8-related imports"""
    print("=== Testing FP8 Imports ===")
    
    try:
        from llm_quant_tool.quant.fp8 import quantise_fp8, HAS_TRANSFORMER_ENGINE
        print("✓ FP8 quantization function available")
        print(f"✓ Transformer Engine available: {'Yes' if HAS_TRANSFORMER_ENGINE else 'No (fallback mode)'}")
        return True
    except ImportError as e:
        print(f"✗ FP8 import failed: {e}")
        return False

def test_fp8_dependencies():
    """Test FP8 dependencies"""
    print("\n=== Testing FP8 Dependencies ===")
    
    deps = ["torch", "transformers", "safetensors"]
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
    
    # Check PyTorch FP8 support
    try:
        if hasattr(torch, 'float8_e4m3fn'):
            print("✓ PyTorch FP8 E4M3 support available")
        else:
            print("⚠️  PyTorch FP8 not available, will use INT8 fallback")
    except Exception as e:
        print(f"✗ Error checking PyTorch FP8: {e}")
    
    return len(missing) == 0

def test_fp8_config_loading():
    """Test FP8 config loading"""
    print("\n=== Testing FP8 Config Loading ===")
    
    try:
        from llm_quant_tool.config import QuantConfig
        
        # Try to load FP8 config
        config_path = Path(__file__).parent.parent / "configs" / "fp8.yaml"
        if config_path.exists():
            cfg = QuantConfig.load(str(config_path))
            print(f"✓ FP8 config loaded: {cfg.model_name_or_path}")
            print(f"  Method: {cfg.method}")
            print(f"  Alpha: {getattr(cfg, 'alpha', 'default')}")
            return True
        else:
            print(f"✗ FP8 config file not found: {config_path}")
            return False
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_transformer_engine_optional():
    """Test transformer-engine optional import"""
    print("\n=== Testing Transformer Engine (Optional) ===")
    
    try:
        import transformer_engine
        # Get version from package metadata since __version__ may not be available
        try:
            version = getattr(transformer_engine, '__version__', 'unknown')
            print(f"✓ transformer-engine available: {version}")
        except:
            print("✓ transformer-engine available: version unknown")
        
        # Test basic import
        try:
            import transformer_engine.pytorch as te
            print("✓ transformer-engine PyTorch backend available")
        except Exception as e:
            print(f"⚠️  transformer-engine PyTorch backend issue: {e}")
            print("ℹ️  FP8 will use pure PyTorch fallback")
        
        return True
    except ImportError:
        print("ℹ️  transformer-engine not installed (optional)")
        print("ℹ️  FP8 quantization will use pure PyTorch implementation")
        print("💡 For acceleration, install: pip install transformer-engine[pytorch]")
        return True  # This is not a failure

def main():
    """Run all FP8 tests"""
    print("FP8 Debug Script")
    print("==============================")
    
    tests = [
        ("FP8 Imports", test_fp8_imports),
        ("FP8 Dependencies", test_fp8_dependencies),
        ("FP8 Config Loading", test_fp8_config_loading),
        ("Transformer Engine (Optional)", test_transformer_engine_optional),
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    print("\n==============================")
    print("FP8 DEBUG SUMMARY")
    print("==============================")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:<30} {status}")
    
    if passed == total:
        print("\n🎉 FP8 is ready to use!")
        print("💡 Tip: Use GPU for better performance")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check output above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
