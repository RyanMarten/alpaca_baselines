#!/bin/bash
#SBATCH --partition=gpu-a100-dev
#SBATCH --job-name=hello-multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH -t 0:30:00
#SBATCH --output=logs/%x_%j_%N_%t.out

echo "Job started at $(date)"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NODELIST: $SLURM_NODELIST"

# Force execution on all allocated nodes
srun --nodes=2 --ntasks=2 hostname
srun --nodes=2 --ntasks=2 echo "Hello world from \$SLURMD_NODENAME (SLURM_PROCID: \$SLURM_PROCID)"

# Optional: Add more commands to demonstrate multi-node execution
srun --nodes=2 --ntasks=2 sleep 2
srun --nodes=2 --ntasks=2 date

echo "Job ended at $(date), reported by $SLURM_NODENAME with PROCID $SLURM_PROCID"

