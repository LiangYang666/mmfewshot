#!/bin/bash

PYTHONPATH="$(dirname $0)/../../":$PYTHONPATH

python tools/misc/print_config.py \
configs/detection/fsce/coco/fsce_r101_fpn_coco_10shot-fine-tuning.py  \
--cfg-options evaluation.interval=2000