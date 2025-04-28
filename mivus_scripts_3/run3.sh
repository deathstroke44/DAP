#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -A PAS2030
#SBATCH --ntasks-per-node=1

milvus_classifier_prior_class_vanila_3.py &> milvus_classifier_prior_class_vanila_3.py.log



