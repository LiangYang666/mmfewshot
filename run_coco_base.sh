#!/bin/bash
echo $0 $1
if [ $1 = "mata" ]
then
  echo "using mata"
  bash ./tools/detection/dist_train.sh \
  configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_base-training.py 2 \
  --cfg-options data.samples_per_gpu=2 data.model_init.samples_per_gpu=4
elif [ $1 = "fsce" ]
then
  echo "using fsce"
  bash ./tools/detection/dist_train.sh \
  configs/detection/fsce/coco/fsce_r101_fpn_coco_base-training.py 2
fi