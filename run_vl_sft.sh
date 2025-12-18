#!/bin/bash

# 启用 ModelScope 下载加速
export USE_MODELSCOPE_HUB=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ENABLE_HF_TRANSFER=0 


# 定义模型路径
MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"

# 定义输出目录
OUTPUT_DIR="saves/qwen3vl8b/sft/mllm_demo"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 启动训练 (SFT - 指令微调模式)
# 特别说明:
# 1. --stage sft: 图文对训练属于指令微调范畴。
# 2. --dataset mllm_demo: 使用 LLaMA-Factory 自带的多模态示例数据。
# 3. --template qwen: 显式指定模版，确保 <image> 标签被正确处理。
# 4. 配置沿用 8B 模型的优化方案 (ZeRO-3 No Offload, FP16, Batch Size 2).

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "开始多模态 SFT 训练..."
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path "$MODEL_PATH" \
    --dataset mllm_demo \
    --finetuning_type lora \
    --lora_target all \
    --output_dir "$OUTPUT_DIR" \
    --template qwen \
    --cutoff_len 4096 \
    --preprocessing_num_workers 40 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_steps 100 \
    --learning_rate 1e-4 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16 \
    --packing false \
    --deepspeed examples/deepspeed/ds_z3_offload_optimized.json

# 注意: packing 对于多模态 SFT 通常设为 false，以免切断图片 token
echo "训练结束。结果保存在: $OUTPUT_DIR"
