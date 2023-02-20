#!/bin/bash

#SBATCH --time=15-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4


DD_PATH=$1
TOKENIZER=$2
MAX_SEQ_LEN=$3
SAVE_PATH=$4

export PYTHONPATH=$PYTHONPATH:/home/mksurkov/LE

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 \
    src/retrieval/tokenize_dd.py \
        --dd-path $DD_PATH \
        --tokenizer $TOKENIZER \
        --max-seq-len $MAX_SEQ_LEN \
        --save-path $SAVE_PATH ;
