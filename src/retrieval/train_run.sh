#!/bin/bash

#SBATCH --time=15-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

TRAINING_CONFIG=$1

export PYTHONPATH=$PYTHONPATH:/home/mksurkov/LE

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 \
    src/retrieval/train.py \
        --training-config $TRAINING_CONFIG ;
