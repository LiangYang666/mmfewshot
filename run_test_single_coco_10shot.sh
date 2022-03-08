#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
PYTHONPATH="$(dirname $0)/../../":$PYTHONPATH

python ./tools/detection/test.py \
/media/E_4TB/YL/mmlab/mmfewshot/configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_xyb_10shot_novel-fine-tuning.py \
/media/E_4TB/YL/mmlab/mmfewshot/work_dirs/meta-rcnn_r50_c4_8xb4_xyb_10shot_novel-fine-tuning/iter_30000.pth \
--cfg-options \
data.samples_per_gpu=12 data.workers_per_gpu=4  \
data.model_init.samples_per_gpu=12 data.model_init.workers_per_gpu=4 \
evaluation.jsonfile_prefix=result/result \
--show-dir result/result_show \
--out result/result.pkl \
--eval bbox \
--show-score-thr 0.1