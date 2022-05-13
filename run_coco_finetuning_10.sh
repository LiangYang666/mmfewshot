#!/bin/bash

echo $0 $1
if [ $1 = "mata" ]
then
  bash ./tools/detection/dist_train.sh \
  configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_10shot-fine-tuning.py 2 \
  --cfg-options data.samples_per_gpu=4 data.model_init.samples_per_gpu=8
elif [ $1 = "fsce" ]
then
  bash ./tools/detection/dist_train.sh \
  configs/detection/fsce/coco/fsce_r101_fpn_coco_10shot-fine-tuning.py 2 \
  --cfg-options load_from='checkpoints/tfa_r101_fpn_coco_base-training_20211102_030413_random-init-bbox-head-ea1c2981.pth'
fi