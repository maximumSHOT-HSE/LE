#!/bin/bash

#SBATCH --time=15-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

DD_PATH=$1
EXP_DIR_PATH=$2
ST_NAME=$3
BATCH_SIZE=$4
NUM_EPOCHS=$5
FULL=$6

export PYTHONPATH=$PYTHONPATH:/home/mksurkov/LE

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 \
    src/retrieval/train_with_st.py \
        --dd-path $DD_PATH \
        --exp-dir-path $EXP_DIR_PATH \
        --st-name $ST_NAME \
        --batch-size $BATCH_SIZE \
        --num-epochs $NUM_EPOCHS \
        --full $FULL ;
