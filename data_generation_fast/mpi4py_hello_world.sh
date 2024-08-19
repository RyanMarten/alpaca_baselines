#!/bin/bash
#SBATCH --partition=development
#SBATCH --job-name=mpi4py_hello_world
#SBATCH --output=../logs/%x_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00

# this does work with #SBATCH --ntasks-per-node=2+
# if you don't provide anything it defaults to number of cores

# Run the MPI program
echo "Job started at $(date)"
ibrun ../venv/bin/python mpi4py_loop_example.py
echo "Job ended at $(date), reported by $SLURMD_NODENAME with PROCID $SLURM_PROCID"

