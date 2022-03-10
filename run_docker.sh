#!/bin/bash

docker run --gpus all -it \
--name mmfewshot \
-v /media/E_4TB/YL/mmlab/mmfewshot/data/few_shot_ann:/mmfewshot/data/few_shot_ann \
-v /media/E_4TB/YL/datasets/xyb/xyb_crop:/mmfewshot/data/xyb \
-v /media/E_4TB/YL/mmlab/mmfewshot/work_dirs:/mmfewshot/work_dirs \
mmfewshot