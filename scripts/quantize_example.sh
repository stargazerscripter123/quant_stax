#!/usr/bin/env bash
python -m llm_quant_tool.cli -c configs/bnb_fp4.yaml --model-type auto
python -m llm_quant_tool.cli -c configs/awq.yaml --model-type auto
python -m llm_quant_tool.cli -c configs/gptq.yaml --model-type auto
python -m llm_quant_tool.cli -c configs/fp8.yaml --model-type auto
python -m llm_quant_tool.cli -c configs/fp8_dynamic.yaml --model-type auto
python -m llm_quant_tool.cli -c configs/nvfp4.yaml --model-type auto