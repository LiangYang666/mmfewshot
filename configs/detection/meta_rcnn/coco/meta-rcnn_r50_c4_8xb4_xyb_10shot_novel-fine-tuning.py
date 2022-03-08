_base_ = [
    '../../_base_/datasets/nway_kshot/few_shot_xyb.py',
    '../../_base_/schedules/schedule.py', '../meta-rcnn_r50_c4.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    train=dict(
        save_dataset=True,
        num_used_support_shots=100,
        dataset=dict(
            type='FewShotXYBDefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting='10SHOT')],
            num_novel_shots=100,
            # num_base_shots=10,
        )),
    model_init=dict(num_novel_shots=100))
evaluation = dict(interval=500, jsonfile_prefix='result/result')
checkpoint_config = dict(interval=500)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None, step=[5000])
runner = dict(max_iters=5000)
# load_from = 'path of base training model'
load_from = \
    'mytools/meta-rcnn_r50_c4_8xb4_coco_base-training_20211102_213915-65a22539.pth'
# model settings
model = dict(frozen_parameters=[
    'backbone', 'shared_head', 'rpn_head', 'aggregation_layer'
])
