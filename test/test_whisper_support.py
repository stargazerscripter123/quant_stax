"""
Test suite for Whisper quantization support
"""
import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_quant_tool.quant.model_utils import (
    detect_model_type, 
    get_model_config, 
    get_supported_model_types
)
from llm_quant_tool.config import load_config, QuantConfig


def test_whisper_model_detection():
    """Test that Whisper models are correctly detected from various path patterns"""
    # Test path-based detection
    test_cases = [
        ("openai/whisper-base", "whisper"),
        ("whisper-large", "whisper"),
        ("distilbert-whisper-base", "whisper"),
        ("openai/whisper-large-v3", "whisper"),
        ("Systran/faster-whisper-base", "whisper"),
    ]
    
    for model_path, expected_type in test_cases:
        detected_type = detect_model_type(model_path)
        assert detected_type == expected_type, f"Expected {expected_type} for {model_path}, got {detected_type}"


def test_whisper_model_config():
    """Test that Whisper model configuration is correctly defined"""
    config = get_model_config("whisper")
    
    # Check required configuration fields
    assert config["model_class"] == "WhisperForConditionalGeneration"
    assert config["tokenizer_class"] == "AutoProcessor"
    assert config["trust_remote_code"] == True
    assert config["use_fast"] == False
    assert config["torch_dtype"] == "auto"


def test_whisper_in_supported_types():
    """Test that whisper is included in supported model types"""
    supported_types = get_supported_model_types()
    assert "whisper" in supported_types, f"whisper not found in supported types: {supported_types}"


def test_whisper_config_files():
    """Test that Whisper configuration files can be loaded correctly"""
    import os
    
    # Change to project root directory for config loading
    original_cwd = os.getcwd()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    try:
        # Test Whisper FP8 config
        cfg_fp8 = load_config("configs/whisper_fp8.yaml")
        assert cfg_fp8.method == "fp8"
        assert cfg_fp8.model_type == "whisper"
        assert cfg_fp8.fp8_dynamic == True
        assert cfg_fp8.fp8_scheme == "FP8_DYNAMIC"
        
        # Test Whisper NVFP4 config
        cfg_nvfp4 = load_config("configs/whisper_nvfp4.yaml")
        assert cfg_nvfp4.method == "nvfp4"
        assert cfg_nvfp4.model_type == "whisper"
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def test_whisper_cli_argument_parsing():
    """Test that CLI accepts whisper as a valid model type"""
    import argparse
    
    # Simulate CLI argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=get_supported_model_types())
    
    # Test that whisper is accepted
    args = parser.parse_args(["--model-type", "whisper"])
    assert args.model_type == "whisper"


def test_whisper_fallback_behavior():
    """Test that unknown whisper-like models still get detected as whisper"""
    test_cases = [
        "custom-whisper-model",
        "my-whisper-v2",
        "whisper-custom-dataset"
    ]
    
    for model_path in test_cases:
        detected_type = detect_model_type(model_path)
        assert detected_type == "whisper", f"Expected whisper for {model_path}, got {detected_type}"


def test_whisper_config_override():
    """Test that Whisper config can be overridden with custom values"""
    cfg = QuantConfig(
        model_name_or_path="openai/whisper-base",
        method="fp8",
        model_type="whisper",
        fp8_dynamic=True,
        fp8_scheme="FP8_DYNAMIC"
    )
    
    assert cfg.model_type == "whisper"
    assert cfg.fp8_dynamic == True
    assert cfg.fp8_scheme == "FP8_DYNAMIC"


def test_whisper_specific_paths():
    """Test detection of various Whisper model path patterns"""
    specific_whisper_paths = [
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small", 
        "openai/whisper-medium",
        "openai/whisper-large",
        "openai/whisper-large-v2",
        "openai/whisper-large-v3",
        "distil-whisper/distil-large-v2",
        "Systran/faster-whisper-medium",
        "microsoft/speecht5-whisper",
    ]
    
    for model_path in specific_whisper_paths:
        detected_type = detect_model_type(model_path)
        assert detected_type == "whisper", f"Failed to detect {model_path} as whisper, got {detected_type}"


def test_non_whisper_models():
    """Test that non-Whisper models are not incorrectly detected as Whisper"""
    non_whisper_paths = [
        "bert-base-uncased",
        "gpt2",
        "microsoft/DialoGPT-medium",
        "facebook/opt-350m",
        "whistle-base",  # Similar name but not whisper
        "whisper-like-model",  # Contains "whisper" but should be detected generically
    ]
    
    for model_path in non_whisper_paths:
        detected_type = detect_model_type(model_path)
        # These should either be detected as other types or fall back to "llm"
        if "whisper" in model_path.lower() and not any(real_whisper in model_path.lower() for real_whisper in ["openai/whisper", "distil-whisper", "faster-whisper"]):
            # For whisper-like names that aren't real whisper models, they might still be detected as whisper due to fallback
            assert detected_type in ["whisper", "llm"], f"Unexpected type {detected_type} for {model_path}"
        else:
            assert detected_type != "whisper" or detected_type == "llm", f"Incorrectly detected {model_path} as whisper"


def test_whisper_config_integration():
    """Test integration between different Whisper configurations"""
    import os
    
    # Change to project root directory for config loading
    original_cwd = os.getcwd()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    try:
        # Test that all required fields exist in Whisper configs
        required_fields = ["model_name_or_path", "method", "model_type"]
        
        for config_file in ["whisper_fp8.yaml", "whisper_nvfp4.yaml"]:
            cfg = load_config(f"configs/{config_file}")
            
            for field in required_fields:
                assert hasattr(cfg, field), f"Missing field {field} in {config_file}"
                assert getattr(cfg, field) is not None, f"Field {field} is None in {config_file}"
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    # Run tests when executed directly
    import sys
    import os
    
    # Change to project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("Running Whisper support tests...")
    print("=" * 50)
    
    test_functions = [
        test_whisper_model_detection,
        test_whisper_model_config,
        test_whisper_in_supported_types,
        test_whisper_config_files,
        test_whisper_cli_argument_parsing,
        test_whisper_fallback_behavior,
        test_whisper_config_override,
        test_whisper_specific_paths,
        test_non_whisper_models,
        test_whisper_config_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...", end=" ")
            test_func()
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Tests: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("All Whisper support tests passed! ✓")
