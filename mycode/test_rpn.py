#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :test_rpn.py
# @Time      :2022/5/15 下午3:46
# @Author    :Yangliang
import torch
from mmdet.models import build_head

rpn_head = dict(
    type='RPNHead',
    in_channels=256,
    feat_channels=256,
    anchor_generator=dict(
        type='AnchorGenerator',
        scales=[8],
        ratios=[0.5, 1.0, 2.0],
        strides=[4, 8, 16, 32, 64]),
    bbox_coder=dict(
        type='DeltaXYWHBBoxCoder',
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0]),
    loss_cls=dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    loss_bbox=dict(type='L1Loss', loss_weight=1.0))

if __name__ == "__main__":
    rpn_head_model = build_head(rpn_head)
    rpn_head_model.eval()
    scales = [8, 4, 2, 1]
    inputs = [torch.rand(1, 256, s, s) for s in scales]
    outputs = rpn_head_model(inputs)

    print(outputs)