#!/bin/bash

# LLaMA-Factory 环境配置脚本
# 配置缓存和存储路径到 /data0 目录

# 设置缓存目录到data盘
export HF_HOME=/data0/tcmofashi/cache/huggingface
export TRANSFORMERS_CACHE=/data0/tcmofashi/cache/transformers
export HF_DATASETS_CACHE=/data0/tcmofashi/cache/datasets

# 设置默认保存目录
export DEFAULT_SAVE_DIR=/data0/tcmofashi/saves
export DEFAULT_LOG_DIR=/data0/tcmofashi/logs
export DEFAULT_TEMP_DIR=/data0/tcmofashi/temp

# 其他HuggingFace配置
export USE_MODELSCOPE_HUB=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "LLaMA-Factory 环境变量已设置:"
echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "DEFAULT_SAVE_DIR: $DEFAULT_SAVE_DIR"