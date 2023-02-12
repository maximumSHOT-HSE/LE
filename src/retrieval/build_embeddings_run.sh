#!/bin/bash

#SBATCH --time=15-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4


DD_PATH=$1
RETRIEVER=$2
SAVE_PATH=$3


module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 \
    /home/mksurkov/LE/src/retrieval/build_embeddings.py \
        --dd-path $DD_PATH \
        --retriever $RETRIEVER \
        --save-path $SAVE_PATH ;
