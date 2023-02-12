#!/bin/bash

#SBATCH --time=15-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4


DD_PATH=$1
TOKENIZER=$2
SAVE_PATH=$3

export PYTHONPATH=$PYTHONPATH:/home/mksurkov/LE

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 \
    src/reranker/tokenize_dd.py \
        --dd-path $DD_PATH \
        --tokenizer $TOKENIZER \
        --save-path $SAVE_PATH ;
