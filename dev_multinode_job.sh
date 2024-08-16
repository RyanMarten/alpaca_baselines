#!/bin/bash
#SBATCH --partition=gpu-a100-dev
#SBATCH --job-name=dev-dcft-alpaca-multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00
#SBATCH --output=logs/%x_%j.out
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

# Get the node rank from SLURM_NODEID
export NODE_RANK=$SLURM_NODEID
export GPUS_PER_NODE=2

export NCCL_DEBUG=info
export TORCH_DISTRIBUTED_DEBUG=DETAIL

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
srun $WORK/dcft/stanford_alpaca/venv/bin/torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \ 
    $WORK/dcft/stanford_alpaca/train.py \
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
