#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=distributed_rouge
#SBATCH --output=../logs/%x_%j.out
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00

# Run the MPI program
echo "Job started at $(date)"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
ibrun ../venv/bin/python -m distributed_rouge filter_rouge --input_file regen.json --output_file filtered.json
echo "Job ended at $(date), reported by $SLURMD_NODENAME with PROCID $SLURM_PROCID"

