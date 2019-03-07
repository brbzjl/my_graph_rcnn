#!/bin/bash

DATASET=$1
NET=$2
IMDB=$3
BATCH_SIZE=$4
NUM_WORKER=$5
SESSION=$6


LOG="logs/${NET}_${IMDB}_${EXTRA_ARGS_SLUG}_${NET}`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./trainval_grcnn.py \
    --dataset ${DATASET} \
    --net ${NET} \
    --imdb ${IMDB} \
    --bs ${BATCH_SIZE} \
    --nworker ${NUM_WORKER} \
    --s ${SESSION} \
    --mGPUs 1 \
    --r True \
    --checksession 10 \
    --checkepoch 4 \
    --checkpoint 10712
