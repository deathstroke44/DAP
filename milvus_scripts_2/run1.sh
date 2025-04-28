#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -A PAS2030
#SBATCH --ntasks-per-node=6

python create_indexes_1.py &> run1.log



