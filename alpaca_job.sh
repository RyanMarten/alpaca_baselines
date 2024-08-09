#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --job-name=dfct-alpaca
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=42
#SBATCH -t 48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --account=CCR23021

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG=info
export NCCL_NET_GDR_LEVEL="SYS"
export NCCL_NET_GDR_READ=1

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

export PATH="$WORK/conda/miniconda3/condabin:$PATH"
source $WORK/conda/miniconda3/etc/profile.d/conda.sh
conda activate dcnlp  # install according to tng/tools/environment.yml

SCALE=$1 
DATA=$2
LOGS=$3
REMOTE_SYNC=$4

# This assumes that each folder has been moved from s3 to the same structure in $SCRATCH
MANIFEST_PREFIX=$(cat $DATA | jq .manifest_url | sed 's|s3://dcnlp-west|${SCRATCH}' | sed 's|s3://dcnlp-hub|${SCRATCH}' | sed 's|\"||g' | sed 's|/manifest.jsonl||')

echo "node-list: $SLURM_JOB_NODELIST"

srun --accel-bind=gn python -m training.train \
    --scale $SCALE \
    --data-config $DATA \
    --logs $LOGS \
    --remote-sync $REMOTE_SYNC \
    --manifest-prefix-override $MANIFEST_PREFIX \
    --clean-exp \
    --report-to-wandb
