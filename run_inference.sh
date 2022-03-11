#!/bin/bash

PYTHONPATH="$(dirname $0)/../../":$PYTHONPATH

python ./tools/detection/inference.py -input ./data/xyb/images/val -output ./result_show \
--checkpoint work_dirs/xyb-rcnn_r50_c4_8xb4_novel-fine-tuning/iter_10000.pth \
--save-support-heatmap \
--save-query-heatmap
