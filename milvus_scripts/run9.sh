#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -A PAS2030
#SBATCH --ntasks-per-node=6

python milvus_classifier_prior_class_4.py &> run9.log



