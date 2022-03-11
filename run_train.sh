#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
PYTHONPATH="$(dirname $0)/../../":$PYTHONPATH

python ./tools/detection/train.py \
mytools/xyb-rcnn_r50_c4_8xb4_novel-fine-tuning.py \
--cfg-options data.samples_per_gpu=12 data.workers_per_gpu=4  \
data.model_init.samples_per_gpu=12 data.model_init.workers_per_gpu=4 \
evaluation.interval=1000 runner.max_iters=10000