#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :test_anchorGenerator.py
# @Time      :2022/5/16 上午10:35
# @Author    :Yangliang


import torch
from mmdet.core import build_prior_generator

anchor_generator_c4 = dict(
    type='AnchorGenerator',
    scales=[2, 4, 8, 16, 32],
    ratios=[0.5, 1.0, 2.0],
    scale_major=False,
    strides=[16])

anchor_generator_fpn = dict(
    type='AnchorGenerator',
    scales=[8],                 # 例如第一个层级的第一个anchor的1.0比例长宽均为8*4
    ratios=[0.5, 1.0, 2.0],
    strides=[4, 8, 16, 32, 64])  # 各尺度特征图相对于原图的步长 下采样率(即生成的坐标位移 )

if __name__ == "__main__":
    anchor_generator_model = build_prior_generator(anchor_generator_fpn)
    all_anchors = anchor_generator_model.grid_priors([(16, 16), (8, 8), (4, 4), (2, 2), (1, 1)], device='cpu')
    print(all_anchors)
    # print(all_anchors[4])
    # tensor([[-362.0387, -181.0193,  362.0387,  181.0193],
    #         [-256.0000, -256.0000,  256.0000,  256.0000],
    #         [-181.0193, -362.0387,  181.0193,  362.0387]])

