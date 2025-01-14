#!/bin/bash
#SBATCH --job-name=hpcg_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:45:00
#SBATCH --output=hpcg_output.txt


LD_PRELOAD=./quantum_shim.so mpirun -n 1 ./xhpcg hpcg.dat
