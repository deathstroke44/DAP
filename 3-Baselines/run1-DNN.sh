#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -A PAS2030
#SBATCH --ntasks-per-node=6

python DNN.py --city Atlanta &> DNN-Atlanta.log



