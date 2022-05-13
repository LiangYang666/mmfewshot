#!/bin/bash
bash ./tools/detection/dist_train.sh \
configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_base-training.py 2 \
--cfg-options data.samples_per_gpu=2 data.model_init.samples_per_gpu=4