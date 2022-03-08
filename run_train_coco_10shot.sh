#!/bin/bash


bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_xyb_crop_10shot-fine-tuning.py 4 \
--cfg-options data.samples_per_gpu=12 data.workers_per_gpu=4  \
data.model_init.samples_per_gpu=12 data.model_init.workers_per_gpu=4 \
evaluation.interval=1000 runner.max_iters=50000