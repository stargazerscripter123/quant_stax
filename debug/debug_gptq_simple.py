#!/usr/bin/env python3
"""
Simple GPTQ debug script to test GPTQModel with minimal setup
Run from debug/ directory: python debug_gptq_simple.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import llm_quant_tool
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_gptq_imports():
    """Test GPTQ-related imports"""
    print("=== Testing GPTQ Imports ===")
    
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
        print("✓ GPTQModel imports successful")
        return True
    except ImportError as e:
        print(f"✗ GPTQModel import failed: {e}")
        return False

def test_simple_gptq_setup():
    """Test basic GPTQ model setup without quantization"""
    print("\n=== Testing Simple GPTQ Setup ===")
    
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
        
        # Use a tiny model for quick testing
        model_name = "distilgpt2"
        print(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded")
        
        # Test quantization config
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=False,
        )
        print("✓ QuantizeConfig created")
        
        # Test model initialization (without actual quantization)
        try:
            model = GPTQModel.from_pretrained(
                model_name,
                quantize_config=quantize_config,
                torch_dtype=torch.float16,
                device_map="cpu"  # Use CPU to avoid GPU issues
            )
            print(f"✓ GPTQModel initialized on CPU")
            
            # Test a simple tokenization
            text = "The quick brown fox jumps over the lazy dog."
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            print(f"✓ Tokenization test successful: {tokens['input_ids'].shape}")
            
            return True
            
        except Exception as e:
            print(f"✗ Model initialization failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ GPTQ setup failed: {e}")
        return False

def test_device_availability():
    """Test CUDA availability for GPTQ"""
    print("\n=== Testing Device Availability ===")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️  CUDA not available, will use CPU (slower)")
        return False

def test_llm_quant_tool_gptq():
    """Test GPTQ through llm_quant_tool"""
    print("\n=== Testing LLM Quant Tool GPTQ Integration ===")
    
    try:
        from llm_quant_tool.quant.gptq import quantise_gptq
        print("✓ GPTQ quantization function available")
        return True
    except ImportError as e:
        print(f"✗ GPTQ integration import failed: {e}")
        return False

def main():
    """Run GPTQ debug tests"""
    print("GPTQ Debug Script\n" + "="*30)
    
    # Run tests
    imports_ok = test_gptq_imports()
    device_ok = test_device_availability()
    setup_ok = test_simple_gptq_setup() if imports_ok else False
    integration_ok = test_llm_quant_tool_gptq()
    
    # Summary
    print("\n" + "="*30)
    print("GPTQ DEBUG SUMMARY")
    print("="*30)
    print(f"Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Device: {'✓ CUDA' if device_ok else '⚠️  CPU'}")
    print(f"Setup: {'✓ PASS' if setup_ok else '✗ FAIL'}")
    print(f"Integration: {'✓ PASS' if integration_ok else '✗ FAIL'}")
    
    if imports_ok and setup_ok and integration_ok:
        print("\n🎉 GPTQ is ready to use!")
        if device_ok:
            print("💡 Tip: Run with GPU for faster quantization")
        return 0
    else:
        print("\n⚠️  Some GPTQ issues found. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
