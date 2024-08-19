#!/bin/bash
#SBATCH --partition=development
#SBATCH --job-name=mpi4py_hello_world
#SBATCH --output=../logs/%x_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:05:00

# Load any necessary modules (adjust as needed for your cluster)
# module load python/3.8
# module load mpi/openmpi-4.0

# Run the MPI program
echo "Job started at $(date)"
srun ../venv/bin/python mpi4py_example.py
echo "Job ended at $(date), reported by $SLURMD_NODENAME with PROCID $SLURM_PROCID"

