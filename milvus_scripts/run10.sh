#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -A PAS2030
#SBATCH --ntasks-per-node=6

python create_index_only_2.py &> create_index_only_2.log && python insert_to_db_collection_6.py &> insert_to_db_collection_6.log && python insert_to_db_collection_7.py &> insert_to_db_collection_7.log && python milvus_classifier_prior_class_6.py &> milvus_classifier_prior_class_6.log && python milvus_classifier_prior_class_7.py &> milvus_classifier_prior_class_7.log && echo 1



