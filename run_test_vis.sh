#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
PYTHONPATH="$(dirname $0)/../../":$PYTHONPATH

python ./tools/detection/test.py \
mytools/xyb_vis-rcnn_r50_c4_8xb4_novel-fine-tuning.py \
checkpoints/xyb_vis-fine-tuning-iter_2000.pth \
-output ./result_show_vis \
--show-score-thr 0.3 \
--save-support-heatmap \
--save-query-heatmap