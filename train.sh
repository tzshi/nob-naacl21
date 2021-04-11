#!/bin/bash

source ~/environments/python3env/bin/activate

TRAIN_STEPS=20000

BATCH_SIZE=8

LEARNING_RATE=1e-5
BETA1=0.9
BETA2=0.999
EPSILON=1e-12

CLIP=1.0
WARMUP=2000

CHART_DIMS=256
CHART_DROP=0.

LOG_FOLDER=./models/

mkdir -p $LOG_FOLDER

DATASOURCE=qasrl
# DATASOURCE=wiki

CHART_MODE=strict
# CHART_MODE=loose

TRAIN_FILE=./data/${DATASOURCE}-train.json

RUN=${DATASOURCE}_${CHART_MODE}
SAVE_PREFIX=${LOG_FOLDER}/${RUN}
mkdir -p $SAVE_PREFIX

TOKENIZERS_PARALLELISM=false \
python -m nob.parser \
    - create-parser --batch-size $BATCH_SIZE \
        --learning-rate $LEARNING_RATE --beta1 $BETA1 --beta2 $BETA2 --epsilon $EPSILON \
        --clip $CLIP --warmup $WARMUP \
        --chart-dims $CHART_DIMS --chart-drop $CHART_DROP \
        --chart-mode $CHART_MODE \
    - train $TRAIN_FILE \
        --train-steps $TRAIN_STEPS \
        --save-prefix $SAVE_PREFIX/ \
    - finish
