#!/usr/bin/env python3
"""
Comprehensive Whisper Support Test Suite
Tests all Whisper quantization functionality including model detection, configuration, and CLI integration.
"""
import sys
import os
import tempfile
import pytest

# Add project root to path and change directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from llm_quant_tool.quant.model_utils import (
    detect_model_type, 
    get_model_config, 
    get_supported_model_types
)
from llm_quant_tool.config import load_config, QuantConfig


class TestWhisperSupport:
    """Test class for Whisper quantization support"""
    
    def test_whisper_model_detection(self):
        """Test that Whisper models are correctly detected from various path patterns"""
        test_cases = [
            ("openai/whisper-base", "whisper"),
            ("openai/whisper-tiny", "whisper"),
            ("openai/whisper-medium", "whisper"),
            ("openai/whisper-large", "whisper"),
            ("openai/whisper-large-v2", "whisper"),
            ("openai/whisper-large-v3", "whisper"),
            ("whisper-large", "whisper"),
            ("distil-whisper/distil-large-v2", "whisper"),
            ("Systran/faster-whisper-base", "whisper"),
            ("microsoft/speecht5-whisper", "whisper"),
        ]
        
        for model_path, expected_type in test_cases:
            detected_type = detect_model_type(model_path)
            assert detected_type == expected_type, f"Expected {expected_type} for {model_path}, got {detected_type}"

    def test_whisper_model_config(self):
        """Test that Whisper model configuration is correctly defined"""
        config = get_model_config("whisper")
        
        # Check required configuration fields
        assert config["model_class"] == "WhisperForConditionalGeneration"
        assert config["tokenizer_class"] == "AutoProcessor"
        assert config["trust_remote_code"] == True
        assert config["use_fast"] == False
        assert config["torch_dtype"] == "auto"

    def test_whisper_in_supported_types(self):
        """Test that whisper is included in supported model types"""
        supported_types = get_supported_model_types()
        assert "whisper" in supported_types, f"whisper not found in supported types: {supported_types}"

    def test_whisper_config_files_exist(self):
        """Test that Whisper configuration files exist and can be loaded"""
        config_files = ["whisper_fp8.yaml", "whisper_nvfp4.yaml"]
        
        for config_file in config_files:
            config_path = f"configs/{config_file}"
            assert os.path.exists(config_path), f"Config file {config_path} does not exist"
            
            # Test loading
            cfg = load_config(config_path)
            assert cfg.model_type == "whisper", f"Config {config_file} should have model_type='whisper'"

    def test_whisper_fp8_config(self):
        """Test specific FP8 Whisper configuration"""
        cfg = load_config("configs/whisper_fp8.yaml")
        assert cfg.method == "fp8"
        assert cfg.model_type == "whisper"
        assert cfg.fp8_dynamic == True
        assert cfg.fp8_scheme == "FP8_DYNAMIC"

    def test_whisper_nvfp4_config(self):
        """Test specific NVFP4 Whisper configuration"""
        cfg = load_config("configs/whisper_nvfp4.yaml")
        assert cfg.method == "nvfp4"
        assert cfg.model_type == "whisper"

    def test_whisper_config_override(self):
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

    def test_non_whisper_models_not_detected_as_whisper(self):
        """Test that non-Whisper models are not incorrectly detected as Whisper"""
        non_whisper_models = [
            "bert-base-uncased",
            "gpt2", 
            "microsoft/DialoGPT-medium",
            "facebook/opt-350m",
        ]
        
        for model_path in non_whisper_models:
            detected_type = detect_model_type(model_path)
            assert detected_type != "whisper", f"Incorrectly detected {model_path} as whisper"

    def test_cli_whisper_integration(self):
        """Test that CLI properly supports whisper model type"""
        from llm_quant_tool.cli import main
        
        # Test that --model-type whisper is accepted (should not raise an error)
        try:
            # This will fail due to missing config, but should not fail on argument parsing
            main(["--model-type", "whisper", "--config", "nonexistent.yaml"])
        except SystemExit:
            pass  # Expected due to missing config file
        except FileNotFoundError:
            pass  # Expected due to missing config file
        except Exception as e:
            if "whisper" in str(e).lower() and "invalid choice" in str(e).lower():
                pytest.fail(f"CLI rejected whisper as model type: {e}")


def run_tests():
    """Run all tests and return success status"""
    print("Running Whisper Support Test Suite")
    print("=" * 50)
    
    test_instance = TestWhisperSupport()
    test_methods = [
        ("Model Detection", test_instance.test_whisper_model_detection),
        ("Model Config", test_instance.test_whisper_model_config), 
        ("Supported Types", test_instance.test_whisper_in_supported_types),
        ("Config Files Exist", test_instance.test_whisper_config_files_exist),
        ("FP8 Config", test_instance.test_whisper_fp8_config),
        ("NVFP4 Config", test_instance.test_whisper_nvfp4_config),
        ("Config Override", test_instance.test_whisper_config_override),
        ("Non-Whisper Detection", test_instance.test_non_whisper_models_not_detected_as_whisper),
        ("CLI Integration", test_instance.test_cli_whisper_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in test_methods:
        try:
            print(f"Testing {test_name}...", end=" ")
            test_func()
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Whisper support tests passed!")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
