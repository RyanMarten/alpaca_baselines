#!/bin/bash
#SBATCH --partition=gpu-a100-dev
#SBATCH --job-name=hello-multinode
#SBATCH --nodes=2  # Specify the number of nodes you want to use
#SBATCH --ntasks-per-node=1
#SBATCH -t 0:30:00
#SBATCH --output=logs/%x_%j_%t.out
#SBATCH --account=CCR23021
#SBATCH --mail-type=all
#SBATCH --mail-user=marten4@illinois.edu

srun echo "Hello world from node $(hostname) with Slurm proc ID $SLURM_PROCID"

# Set the master node's address (usually the first node in the allocation)
#export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#export MASTER_PORT=29500
#
#echo "Show hostnames $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"
#
#export NODE_RANK=$SLURM_NODEID
#export SLURM_GPUS_PER_NODE=2
#
#echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
#echo "SLURM_NNODES: $SLURM_NNODES"
#echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
#echo "SLURM_JOB_ID: $SLURM_JOB_ID"
#echo "SLURM_PROCID: $SLURM_PROCID"
#echo "MASTER_ADDR: $MASTER_ADDR"
#echo "MASTER_PORT: $MASTER_PORT"
#
#echo "Node: $(hostname)"
