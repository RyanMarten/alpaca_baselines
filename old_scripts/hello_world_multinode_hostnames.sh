#!/bin/bash
#SBATCH --partition=development
#SBATCH --job-name=hello-multinode
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH -t 0:30:00
#SBATCH --output=logs/%x_%j.out


# Set up environment variables for multinode training

# This is Eric's setup - which worked for torchrun
#nodes=($(scontrol show hostname $SLURM_NODELIST))
#nodes_array=($nodes)
#head_node=${nodes_array[0]}
#export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
#export MASTER_PORT=25678
#export NODE_RANK=$SLURM_NODEID

# NOTE: This does NOT work for torchrun on tacc (this is name not ipaddress and for some reason doesn't resolve - didn't look further)
#export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# My new setup...
# I think this is all you need to do since these batch script commands ill only be run once and on the head node
# from https://slurm.schedmd.com/sbatch.html#SECTION_DESCRIPTION
# "When the job allocation is finally granted for the batch script, Slurm runs a single copy of the batch script on the first node in the set of allocated nodes."
export MASTER_ADDR=$(hostname --ip-address)
export MASTER_PORT=25678

echo "Job started at $(date)"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"


echo "Hostnames as seeen on the node running the batch script (first node in the set of allocated)"
echo "$(scontrol show hostname $SLURM_JOB_NODELIST)"
echo "Now doing commands with srun - should run on each node in the list"

# Force execution on all allocated nodes
srun bash -c 'echo "Hello world from $(hostname) (SLURM_PROCID: $SLURM_PROCID). I see MASTER_ADD as $MASTER_ADDR"'

# Optional: Add more commands to demonstrate multi-node execution
srun sleep 2
srun date

echo "Job ended at $(date), reported by $SLURMD_NODENAME with PROCID $SLURM_PROCID"

