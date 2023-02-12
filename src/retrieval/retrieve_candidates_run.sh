#!/bin/bash

#SBATCH --time=15-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4


DD_PATH=$1
N_NEIGHBORS=$2
SAVE_PATH=$3

export PYTHONPATH=$PYTHONPATH:/home/mksurkov/LE

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 \
    src/retrieval/retrieve_candidates.py \
        --dd-path $DD_PATH \
        --n-neighbors $N_NEIGHBORS \
        --save-path $SAVE_PATH ;
