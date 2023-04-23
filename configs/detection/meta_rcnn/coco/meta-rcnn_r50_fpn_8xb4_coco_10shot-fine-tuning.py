_base_ = [
    '../../_base_/datasets/nway_kshot/few_shot_coco.py',
    '../../_base_/schedules/schedule.py', '../meta-rcnn_r50_fpn.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    train=dict(
        save_dataset=True,
        num_used_support_shots=10,
        dataset=dict(
            type='FewShotCocoDefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting='10SHOT')],
            num_novel_shots=10,
            num_base_shots=10,
        )),
    model_init=dict(num_novel_shots=10, num_base_shots=10))
evaluation = dict(interval=1000)
checkpoint_config = dict(interval=1000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None, step=[15000])
runner = dict(max_iters=15000)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/meta-rcnn_r50_fpn_8xb4_coco_base-training/iter_120000.pth'
# model settings
model = dict(frozen_parameters=[
    'backbone', 'neck', 'shared_head', 'rpn_head', 'aggregation_layer'
])
