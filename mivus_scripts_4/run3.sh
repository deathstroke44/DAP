#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -A PAS2030
#SBATCH --ntasks-per-node=6

python insert_to_db_collection_2.py &> s2.log ; insert_to_db_collection_3.py &> s3.log  ; insert_to_db_collection_4.py &> s4.log  ; insert_to_db_collection_5.py &> s5.log 



