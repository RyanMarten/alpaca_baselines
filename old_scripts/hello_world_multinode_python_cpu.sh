#!/bin/bash
#SBATCH --partition=development
#SBATCH --job-name=hello-multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH -t 0:01:00
#SBATCH --output=logs/%x_%j.out

echo "Job started at $(date)"
env | grep SLURM

srun bash -c 'echo "Hello world in bash from $(hostname) (SLURM_PROCID: $SLURM_PROCID)"'

srun python << EOF
import os
node = os.getenv("SLURMD_NODENAME", "unk")
proc_id = os.getenv("SLURM_PROCID", "unk")
print(f"Hello world in python from {node} (SLURM_PROCID: {proc_id})")
EOF

echo "Job ended at $(date), reported by $SLURMD_NODENAME with PROCID $SLURM_PROCID"

