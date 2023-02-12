#!/bin/bash

#SBATCH --time=15-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4


DD_PATH=$1
TOPICS_CSV_PATH=$2
CONTENT_CSV_PATH=$3
SAVE_PATH=$4


export PYTHONPATH=$PYTHONPATH:/home/mksurkov/LE

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 \
    src/filter_out_different_langs.py \
        --dd-path $DD_PATH \
        --topics-csv-path $TOPICS_CSV_PATH \
        --content-csv-path $CONTENT_CSV_PATH \
        --save-path $SAVE_PATH ;
