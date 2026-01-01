#!/bin/bash

# H20 Server - Full Parameter Training Script
# 显存充足 (141GB x 4)，直接进行全量微调 (Full Fine-tuning)
# 相比 LoRA，全量微调能获得更好的模型上限，但需要更多显存 (H20 完全够用)。

export USE_MODELSCOPE_HUB=0
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ENABLE_HF_TRANSFER=0

# 设置缓存目录到data盘
export HF_HOME=/data0/tcmofashi/cache/huggingface
export TRANSFORMERS_CACHE=/data0/tcmofashi/cache/transformers
export HF_DATASETS_CACHE=/data0/tcmofashi/cache/datasets 

MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
OUTPUT_DIR="/data0/tcmofashi/saves/qwen3-30b/cpt/h20_full_finetune"

mkdir -p $OUTPUT_DIR

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# 使用 CUDA 12.6
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "开始 H20 全量参数微调 (Full FT) - 30B Model..."
echo "CUDA 版本: $(nvcc -V | grep 'release' | awk '{print $6}')"
echo "注意: 启用 ZeRO-3 Offload 以支持 8192 长度。学习率 1e-5。"

CUDA_VISIBLE_DEVICES=4,5,6,7 llamafactory-cli train \
    --stage pt \
    --do_train \
    --model_name_or_path "$MODEL_PATH" \
    --dataset light_novel_5000,wikipedia_zh,moe_girl_wiki \
    --finetuning_type full \
    --output_dir "$OUTPUT_DIR" \
    --cutoff_len 8192 \
    --preprocessing_num_workers 40 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_steps 100 \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16 \
    --packing true \
    --deepspeed examples/deepspeed/ds_z3_offload_config.json

echo "训练结束。结果保存在: $OUTPUT_DIR"
