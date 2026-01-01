#!/bin/bash

# H20 Server - Ultimate Performance CPT Script (FP8 + Liger Kernel + ZeRO-3)
# 核心思路:
# 1. FP8: 显存减半，计算翻倍 (H20 Tensor Cores)。
# 2. Liger Kernel: 进一步节省显存，优化 Norm/Loss 计算。
# 3. ZeRO-3 (No Offload): 数据全在 GPU，消除通信瓶颈。
# 4. 速度预估: 理论上的最快配置。

export USE_MODELSCOPE_HUB=0
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ENABLE_HF_TRANSFER=0

# 设置缓存目录到data盘
export HF_HOME=/data0/tcmofashi/cache/huggingface
export TRANSFORMERS_CACHE=/data0/tcmofashi/cache/transformers
export HF_DATASETS_CACHE=/data0/tcmofashi/cache/datasets 

MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
OUTPUT_DIR="/data0/tcmofashi/saves/qwen3-30b/cpt/h20_liger_fast"

mkdir -p $OUTPUT_DIR

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# 增加超时以防万一
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 核心改动:
# 1. Context 降为 4096: 腾出巨大显存空间。
# 2. 回归 ZeRO-3 (No Offload): 既然显存够了，就彻底抛弃 CPU Offload，速度直接起飞。
# 3. Batch Size 4: 保证高吞吐。
# 4. FP8 + Liger: 计算加速双保险。

CUDA_VISIBLE_DEVICES=4,5,6,7 llamafactory-cli train \
    --stage pt \
    --do_train \
    --model_name_or_path "$MODEL_PATH" \
    --dataset light_novel_5000,moe_girl_wiki,c4_demo \
    --finetuning_type full \
    --output_dir "$OUTPUT_DIR" \
    --cutoff_len 4096 \
    --preprocessing_num_workers 40 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --optim adamw_bnb_8bit \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_steps 100 \
    --save_total_limit 3 \
    --resume_from_checkpoint "$OUTPUT_DIR/checkpoint-2300" \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp8 \
    --fp8_backend te \
    --enable_liger_kernel \
    --packing true \
    --ddp_timeout 180000000 \
    --deepspeed examples/deepspeed/ds_z3_config.json

echo "训练结束。结果保存在: $OUTPUT_DIR"
