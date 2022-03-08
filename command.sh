bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_base-training.py 4 


bash ./tools/detection/dist_train.sh \
configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_base-training.py 4 \
--cfg-options data.samples_per_gpu=2 data.model_init.samples_per_gpu=4

bash ./tools/detection/dist_train.sh \
configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_base-training.py 4 \
--cfg-options data.samples_per_gpu=2 data.model_init.samples_per_gpu=4



python tools/misc/print_config.py \
configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_base-training.py \
--cfg-options data.samples_per_gpu=2 data.model_init.samples_per_gpu=4



nohup bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_base-training.py 4 > run.log 2>&1 &



nohup bash ./tools/detection/dist_train.sh \
configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_10shot-fine-tuning.py 4 \
--cfg-options data.samples_per_gpu=2 data.model_init.samples_per_gpu=4 \
> run.log 2>&1 &