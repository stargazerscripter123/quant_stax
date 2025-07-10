#!/usr/bin/env python3
"""
Run all debug scripts for comprehensive testing
Usage: python debug_run_all_tests.py
"""

import sys
import subprocess
from pathlib import Path

def run_script(script_name, description):
    """Run a debug script and return success status"""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"Script: {script_name}")
    print('='*60)
    
    try:
        result = subprocess.run([
            sys.executable, script_name
        ], cwd=Path(__file__).parent, timeout=300)  # 5 minute timeout
        
        success = result.returncode == 0
        print(f"\n{description}: {'✅ PASSED' if success else '❌ FAILED'}")
        return success
        
    except subprocess.TimeoutExpired:
        print(f"\n{description}: ⏰ TIMEOUT (took too long)")
        return False
    except Exception as e:
        print(f"\n{description}: ❌ ERROR - {e}")
        return False

def main():
    """Run all debug tests"""
    print("LLM Quant Tool - Complete Debug Test Suite")
    print("="*60)
    
    tests = [
        ("debug_validate_install.py", "Installation Validation"),
        ("debug_gptq_simple.py", "GPTQ Simple Test"),
        ("debug_awq_api.py", "AWQ API Test"),
        ("debug_fp8.py", "FP8 API Test"),
        ("debug_fp8_dynamic.py", "Dynamic FP8 API Test"),
        ("debug_nvfp4.py", "NVFP4A16 API Test"),
    ]
    
    results = []
    for script, description in tests:
        success = run_script(script, description)
        results.append((description, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("COMPLETE TEST SUITE SUMMARY")
    print('='*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{description:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! LLM Quant Tool is fully functional.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
