#!/bin/bash

#SBATCH --time=15-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

export PYTHONPATH=$PYTHONPATH:/home/mksurkov/LE

RERANKER_PATH=$1
DS_PATH=$2
SAVE_PATH=$3

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 \
    src/reranker/predict_proba.py \
        --reranker-path $RERANKER_PATH \
        --ds-path $DS_PATH \
        --save-path $SAVE_PATH ;
