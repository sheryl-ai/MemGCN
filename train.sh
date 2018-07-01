#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=4:00:00
#PBS -N session1_default
#PBS -A course
#PBS -q ShortQ

#cd $PBS_O_WORKDIR
#Graph Convolutional Networks
declare -a view=("dti_fact")

# 'method', 'data_type', 'support K', 'fdim', 'nhops', 'mem_size', 'code_size', 'n_words', 'edim', 'n_epoch', 'batch_size'
python3 train.py "Motor" "MemGCN" "dti_fact" "in" 30 32 3 12 46 223 32 20 32

echo All done
