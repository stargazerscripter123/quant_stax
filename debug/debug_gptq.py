#!/usr/bin/env python3
"""
Complete GPTQ debug script with actual quantization test
Run from debug/ directory: python debug_gptq.py
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

def test_full_gptq_workflow():
    """Test complete GPTQ workflow with actual config"""
    print("=== Testing Full GPTQ Workflow ===")
    
    try:
        from llm_quant_tool.config import QuantConfig
        from llm_quant_tool.quant.gptq import quantise_gptq
        import yaml
        
        # Load GPTQ config
        config_path = Path(__file__).parent.parent / "configs" / "gptq.yaml"
        if not config_path.exists():
            print("✗ GPTQ config file not found")
            return False
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create a test config with minimal settings
        test_config = {
            **config_data,
            "num_calibration_samples": 4,  # Minimal for testing
            "block_size": 128,            # Smaller block size
            "out_dir": str(Path(__file__).parent / "test_output")
        }
        
        cfg = QuantConfig(**test_config)
        print(f"✓ Config loaded: {cfg.model_name_or_path}")
        
        # Test the quantization (this will take some time)
        print("⚠️  Running actual GPTQ quantization test (this may take a few minutes)...")
        print("   This will use minimal settings for testing")
        
        try:
            result_path = quantise_gptq(cfg)
            print(f"✓ GPTQ quantization completed: {result_path}")
            
            # Check if output file exists
            if result_path.exists():
                size_mb = result_path.stat().st_size / (1024 * 1024)
                print(f"✓ Output file size: {size_mb:.1f} MB")
                return True
            else:
                print("✗ Output file not created")
                return False
                
        except Exception as e:
            print(f"✗ GPTQ quantization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ GPTQ workflow test failed: {e}")
        return False

def test_gptq_memory_usage():
    """Test GPTQ memory requirements"""
    print("\n=== Testing GPTQ Memory Usage ===")
    
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            cached_memory = torch.cuda.memory_reserved(device)
            
            print(f"GPU: {torch.cuda.get_device_name(device)}")
            print(f"Total memory: {total_memory / 1e9:.1f} GB")
            print(f"Allocated: {allocated_memory / 1e9:.3f} GB")
            print(f"Cached: {cached_memory / 1e9:.3f} GB")
            print(f"Free: {(total_memory - cached_memory) / 1e9:.1f} GB")
            
            # Check if we have enough memory for GPTQ
            free_gb = (total_memory - cached_memory) / 1e9
            if free_gb > 2.0:
                print("✓ Sufficient GPU memory for GPTQ")
                return True
            else:
                print("⚠️  Low GPU memory, GPTQ may fail or be slow")
                return False
                
        except Exception as e:
            print(f"✗ Memory check failed: {e}")
            return False
    else:
        print("⚠️  No CUDA available, GPTQ will use CPU (very slow)")
        return False

def test_gptq_cli_integration():
    """Test GPTQ through CLI"""
    print("\n=== Testing GPTQ CLI Integration ===")
    
    try:
        import subprocess
        import tempfile
        
        # Create a minimal test config
        test_config = {
            "model_name_or_path": "distilgpt2",
            "method": "gptq",
            "dataset_name": "brando/small-c4-dataset",
            "num_calibration_samples": 2,
            "block_size": 64,
            "gptq_bits": 4,
            "gptq_group_size": 128,
            "gptq_desc_act": False,
            "out_dir": str(Path(__file__).parent / "test_cli_output"),
            "seed": 42
        }
        
        # Write temporary config
        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_config = f.name
        
        try:
            # Test CLI dry run (help)
            result = subprocess.run([
                sys.executable, "-m", "llm_quant_tool.cli", "--help"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            if result.returncode == 0:
                print("✓ CLI accessible")
                
                # Clean up temp file
                Path(temp_config).unlink()
                return True
            else:
                print(f"✗ CLI test failed: {result.stderr}")
                return False
                
        finally:
            # Clean up temp file
            if Path(temp_config).exists():
                Path(temp_config).unlink()
        
    except Exception as e:
        print(f"✗ CLI integration test failed: {e}")
        return False

def main():
    """Run comprehensive GPTQ debug tests"""
    print("GPTQ Complete Debug Script\n" + "="*40)
    
    # Run tests
    memory_ok = test_gptq_memory_usage()
    cli_ok = test_gptq_cli_integration()
    
    print(f"\n{'='*40}")
    print("Would you like to run the full GPTQ quantization test?")
    print("This will actually quantize a small model and may take 2-5 minutes.")
    print("Enter 'y' to proceed, any other key to skip:")
    
    try:
        user_input = input().strip().lower()
        if user_input == 'y':
            workflow_ok = test_full_gptq_workflow()
        else:
            print("Skipping full workflow test")
            workflow_ok = None
    except KeyboardInterrupt:
        print("\nTest cancelled by user")
        workflow_ok = False
    
    # Summary
    print(f"\n{'='*40}")
    print("GPTQ COMPLETE DEBUG SUMMARY")
    print("="*40)
    print(f"Memory check: {'✓ PASS' if memory_ok else '⚠️  WARN'}")
    print(f"CLI integration: {'✓ PASS' if cli_ok else '✗ FAIL'}")
    if workflow_ok is not None:
        print(f"Full workflow: {'✓ PASS' if workflow_ok else '✗ FAIL'}")
    else:
        print("Full workflow: SKIPPED")
    
    if cli_ok and (workflow_ok is None or workflow_ok):
        print("\n🎉 GPTQ debugging complete!")
        return 0
    else:
        print("\n⚠️  Some GPTQ issues found.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
