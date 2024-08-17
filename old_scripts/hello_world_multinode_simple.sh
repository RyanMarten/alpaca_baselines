#!/bin/bash
#SBATCH --partition=development
#SBATCH --job-name=hello-multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH -t 0:30:00
#SBATCH --output=logs/%x_%j.out

echo "Job started at $(date)"
#echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
#echo "SLURM_NODELIST: $SLURM_NODELIST"
#echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
env | grep SLURM

#this produces two hello worlds, as expected
#srun echo "hello world"

srun hostname
srun bash -c 'echo "Hello world from $(hostname) (SLURM_PROCID: $SLURM_PROCID)"'

#srun sleep 2
#srun date

echo "Job ended at $(date), reported by $SLURMD_NODENAME with PROCID $SLURM_PROCID"

