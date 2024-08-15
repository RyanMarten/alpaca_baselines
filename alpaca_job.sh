#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --job-name=dfct-alpaca
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --account=CCR23021
#SBATCH --mail-type=all
#SBATCH --mail-user=marten4@illinois.edu


# cpu-per-task=0 means use all cpus on the node

$WORK/dcft/stanford_alpaca/venv/bin/torchrun --nproc_per_node=auto --master_port=12345 train.py \
    --model_name_or_path $WORK/dcft/llama-7b \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir $WORK/dcft/llama-7b-checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
