#!/bin/bash
#SBATCH --job-name=my_mpi_job
#SBATCH --output=output2.txt
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH -p debug
#SBATCH --time=00:10:00
mpirun python3 ./timepde/multi_gpu/class_run.py