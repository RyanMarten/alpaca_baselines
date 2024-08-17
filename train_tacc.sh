#!/bin/bash
#SBATCH --partition=gpu-a100-dev
#SBATCH --job-name=dcft-alpaca-reproduction-tacc-8xA100-40GB
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --account=CCR23021

# Job info
echo "Job started at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"

# Networking between nodes to master node

# From srun docs: "Slurm runs a single copy of the batch script on the first node in the set of allocated nodes"
# This first node will be the master node and we will set an environment variable for all the nodes to use
# Therefore this will show PROCID 0 and the ip address of the master node
export MASTER_ADDR=$(hostname --ip-address)
export MASTER_PORT=25678
echo "SLURM_PROCID: $SLURM_PROCID"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# Each node in the list should print their own process id (used as node_rank) and the same master networking info
echo "All nodes Hostnames"
echo "$(scontrol show hostname $SLURM_JOB_NODELIST)"
srun bash -c 'echo "Hello world from $(hostname) (SLURM_PROCID: $SLURM_PROCID). I see MASTER_ADDR:MASTER_PORT as $MASTER_ADDR:$MASTER_PORT"'

# Additional environment variables
export GPUS_PER_NODE=2
echo "GPUS_PER_NODE: $GPUS_PER_NODE"

# Ensure torch and cuda and devices are available
srun $WORK/dcft/stanford_alpaca/venv/bin/python -c "import torch; print('Torch version:', torch.__version__)"
srun $WORK/dcft/stanford_alpaca/venv/bin/python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

# Run the training script
srun echo "Starting torchrun command..."
srun ./venv/bin/torchrun --rdzv-id=$SLURM_JOB_ID --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT --master_addr=$MASTER_ADDR --node_rank=$SLURM_PROCID --nnodes=$SLURM_NNODES --nproc_per_node=auto --master_port=$MASTER_PORT train.py \
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

echo "Job ended at $(date)"
