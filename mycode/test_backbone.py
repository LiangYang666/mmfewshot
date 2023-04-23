#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :test_backbone.py
# @Time      :2022/5/15 下午3:54
# @Author    :Yangliang

import torch
# from mmdet.models import build_backbone
from mmfewshot.detection.models import build_backbone

backbone_c4 = dict(
    type='ResNet',
    depth=50,
    num_stages=3,   # 设为3个stage 原始为4个
    strides=(1, 2, 2),  # 默认为4个stage 改为3个后应重设为长度为3的strides
    dilations=(1, 1, 1),    # 同上
    out_indices=(2,),   # 0为layer1输出 2为layer3输出 原始共4个stage 即输出倒数第二层的特征
    frozen_stages=1,    # 冻结stem和第一个stage 如果是0 冻结stem 如果是-1 冻结所有
    norm_cfg=dict(type='BN', requires_grad=False),
    norm_eval=True,
    style='caffe')

backbone_fpn = dict(
    type='ResNet',
    depth=101,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=False),
    norm_eval=True,
    style='caffe')

backbone_metarcnn =dict(
    type='ResNetWithMetaConv',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=2,
    norm_cfg=dict(type='BN', requires_grad=False),
    norm_eval=True,
    style='caffe')



if __name__ == "__main__":
    backbone_metarcnn_model = build_backbone(backbone_metarcnn)
    backbone_c4_model = build_backbone(backbone_c4)
    backbone_fpn_model = build_backbone(backbone_fpn)
    backbone_c4_model.eval()
    backbone_fpn_model.eval()
    inputs = torch.randn(1, 3, 224, 224)
    outputs_c4 = backbone_c4_model(inputs)
    outputs_fpn = backbone_fpn_model(inputs)
    print(outputs_c4.shape)

    print(outputs_fpn)
    # input (1, 3, 224, 224)
    # for backbone_fpn_model
    # 0: 1, 256, 56, 56
    # 1: 1, 512, 28, 28
    # 2: 1, 1024, 14, 14
    # 3: 1, 2048, 7, 7