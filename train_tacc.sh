#!/bin/bash
#SBATCH --partition=gpu-a100-dev
#SBATCH --job-name=dcft-alpaca-reproduction-tacc-8xA100-40GB
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --account=CCR23021

# Set up environment variables for multinode training
nodes=($(scontrol show hostname $SLURM_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[0]}
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export MASTER_PORT=25678
export NODE_RANK=$SLURM_NODEID
export GPUS_PER_NODE=2

# Debug output
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "hostname on node running the sbatch: $(hostname)"
echo "all hostnames $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"
srun bash -c 'echo "Hello world from $SLURMD_NODENAME (SLURM_PROCID: $SLURM_PROCID) (SLURM_NODEID: $SLURM_NODEID)"'

# ensuring the environment is available
srun $WORK/dcft/stanford_alpaca/venv/bin/python -c "import torch; print('Torch version:', torch.__version__)"
srun $WORK/dcft/stanford_alpaca/venv/bin/python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

# Run the training script
srun echo "Starting torchrun command..."
srun ./venv/bin/torchrun --rdzv-id=$SLURM_JOB_ID --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT --master_addr=$MASTER_ADDR --node_rank=$SLURM_PROCID --nnodes=$SLURM_NNODES --nproc_per_node=auto --master_port=12345 train.py \
    --model_name_or_path $WORK/dcft/llama-7b \
    --data_path ./alpaca_data.json \
    --bf16 False \
    --gradient_checkpointing True \
    --output_dir $WORK/dcft/llama-7b-checkpoints \
    --include_num_input_tokens_seen \
    --include_tokens_per_second \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
