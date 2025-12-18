#!/bin/bash

# 启用 ModelScope 下载加速 (注意：对于非 ModelScope 数据集，需通过 HF 镜像下载)
export USE_MODELSCOPE_HUB=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ENABLE_HF_TRANSFER=0 


# 定义模型路径 (根据实际情况修改)
# Qwen30a3b 可能是指 Qwen2.5-32B 或者特定版本，请在此处填写正确的本地路径或 HuggingFace ID
MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"

# 定义输出目录
OUTPUT_DIR="saves/qwen3vl8b/cpt/light_novel"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 启动训练
# 注意事项：
# 1. --cutoff_len 8192: 适合小说长文本，需要较大显存。显存不足可尝试降低到 4096。
# 2. --packing true: 开启 packing 显著提升长文本预训练效率。
# 3. --per_device_train_batch_size 1: 30B 模型显存占用大，batch size 设为 1，通过 accumulation steps 累积梯度。

export http_proxy=http://10.42.0.17:7890
export https_proxy=http://10.42.0.17:7890

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "开始多模态持续预训练 (CPT) ..."
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train \
    --stage pt \
    --do_train \
    --model_name_or_path "$MODEL_PATH" \
    --dataset light_novel_5000,wikipedia_zh,moe_girl_wiki \
    --finetuning_type lora \
    --lora_target all \
    --output_dir "$OUTPUT_DIR" \
    --cutoff_len 4096 \
    --preprocessing_num_workers 40 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16 \
    --packing true \
    --deepspeed examples/deepspeed/ds_z3_offload_optimized.json

echo "训练结束。结果保存在: $OUTPUT_DIR"
