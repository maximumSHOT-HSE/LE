#!/bin/bash

#SBATCH --time=15-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4


DD_PATH=$1
CORRELATIONS_CSV_PATH=$2
N_NEIGHBORS=$3

export PYTHONPATH=$PYTHONPATH:/home/mksurkov/LE

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 \
    src/retrieval/validate.py \
        --dd-path $DD_PATH \
        --correlations-csv-path $CORRELATIONS_CSV_PATH \
        --n-neighbors $N_NEIGHBORS ;
