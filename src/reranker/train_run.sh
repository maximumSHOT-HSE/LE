#!/bin/bash

#SBATCH --time=15-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

MODEL=$1
TOKENIZER=$2
DD_PATH=$3
DIR_PATH=$4

export PYTHONPATH=$PYTHONPATH:/home/mksurkov/LE

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 \
    src/reranker/train.py \
        --model $MODEL \
        --tokenizer $TOKENIZER \
        --dd-path $DD_PATH \
        --dir-path $DIR_PATH ;
