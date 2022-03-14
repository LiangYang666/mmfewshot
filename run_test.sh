#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
PYTHONPATH="$(dirname $0)/../../":$PYTHONPATH

python ./tools/detection/test.py \
mytools/xyb-rcnn_r50_c4_8xb4_novel-fine-tuning.py \
checkpoints/xyb-fine-tuning-iter_2000.pth \
--cfg-options \
data.samples_per_gpu=12 data.workers_per_gpu=4  \
data.model_init.samples_per_gpu=12 data.model_init.workers_per_gpu=4 \
evaluation.jsonfile_prefix=result/result \
--show-dir result/result_show \
--out result/result.pkl \
--eval bbox \
--show-score-thr 0.1