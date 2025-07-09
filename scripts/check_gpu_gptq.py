#!/usr/bin/env python3
"""
GPU verification script for GPTQ quantization.
Run this to check if GPTQModel is using GPU correctly.
"""

import torch
import logging

def check_gpu_setup():
    """Check GPU availability and setup for GPTQ."""
    logging.basicConfig(level=logging.INFO)
    
    print("=== GPU Setup Check for GPTQ ===")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(i)
            print(f"✅ GPU {i}: {gpu.name}")
            print(f"   Memory: {gpu.total_memory / 1024**3:.1f} GB")
            print(f"   Compute Capability: {gpu.major}.{gpu.minor}")
    else:
        print("❌ CUDA not available")
        print("   Reasons this might happen:")
        print("   - PyTorch not compiled with CUDA support")
        print("   - CUDA drivers not installed")
        print("   - No NVIDIA GPU present")
        return False
    
    # Test GPTQ imports
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
        print("✅ GPTQModel imported successfully")
    except ImportError as e:
        print(f"❌ GPTQModel import failed: {e}")
        return False
    
    # Test basic GPU operations
    try:
        test_tensor = torch.randn(10, 10).cuda()
        print("✅ GPU tensor operations working")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ GPU tensor operations failed: {e}")
        return False
    
    print("\n🎉 GPU setup looks good for GPTQ quantization!")
    return True

if __name__ == "__main__":
    check_gpu_setup()
