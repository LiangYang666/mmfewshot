#!/bin/bash

PYTHONPATH="$(dirname $0)/../../":$PYTHONPATH

python ./tools/detection/inference.py -input ./data/xyb/images/val -output ./result_show \
--checkpoint  checkpoints/xyb-fine-tuning-iter_2000.pth \
--save-support-heatmap \
--save-query-heatmap
