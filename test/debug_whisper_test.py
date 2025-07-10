#!/usr/bin/env python3
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_functionality():
    print("Starting basic functionality test...")
    
    try:
        from llm_quant_tool.quant.model_utils import detect_model_type
        print("✓ Successfully imported detect_model_type")
        
        result = detect_model_type("openai/whisper-base")
        print(f"✓ detect_model_type('openai/whisper-base') = '{result}'")
        
        if result == "whisper":
            print("✓ Whisper detection working correctly")
            return True
        else:
            print(f"✗ Expected 'whisper', got '{result}'")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    print(f"Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
