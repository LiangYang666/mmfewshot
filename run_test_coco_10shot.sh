#!/bin/bash


bash ./tools/detection/dist_test.sh configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_xyb_crop_10shot-fine-tuning.py \
/media/E_4TB/YL/mmlab/mmfewshot/work_dirs/meta-rcnn_r50_c4_8xb4_coco_xyb_crop_10shot-fine-tuning/iter_5000.pth \
4 \
--cfg-options \
data.samples_per_gpu=12 data.workers_per_gpu=4  \
data.model_init.samples_per_gpu=12 data.model_init.workers_per_gpu=4 \
evaluation.jsonfile_prefix=result/result \
--show-dir results_show \
--out result/result.pkl \
--eval bbox
