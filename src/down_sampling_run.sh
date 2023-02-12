#!/bin/bash

#SBATCH --time=15-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

DD_PATH=$1
RATIO=$2
SAVE_PATH=$3

export PYTHONPATH=$PYTHONPATH:/home/mksurkov/LE

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 \
    src/down_sampling.py \
        --dd-path $DD_PATH \
        --ratio $RATIO \
        --save-path $SAVE_PATH ;
