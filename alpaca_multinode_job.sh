#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --job-name=dfct-alpaca-multinode
#SBATCH --nodes=2  # Specify the number of nodes you want to use
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --account=CCR23021
#SBATCH --mail-type=all
#SBATCH --mail-user=marten4@illinois.edu

# Load any necessary modules or activate your virtual environment here
# For example:
# module load cuda/11.7
# source $WORK/dcft/stanford_alpaca/venv/bin/activate

# Set the master node's address (usually the first node in the allocation)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Calculate the total number of processes
export WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))

# Get the node rank from SLURM_NODEID
export NODE_RANK=$SLURM_NODEID

# Run the training script
srun $WORK/dcft/stanford_alpaca/venv/bin/torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
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
