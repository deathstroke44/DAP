#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -A PAS2030
#SBATCH --ntasks-per-node=1

milvus_classifier_prior_class_vanila_2.py &> milvus_classifier_prior_class_vanila_2.py.log



