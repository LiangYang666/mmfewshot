#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :testfpn.py
# @Time      :2022/5/13 下午10:35
# @Author    :Yangliang
import torch
from mmdet.models import build_neck

neck=dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],  #
    out_channels=256,                    #
    start_level=1,                       #
    add_extra_convs='on_input',
    num_outs=4)                          #


if __name__ == "__main__":
    fpn = build_neck(neck).eval()
    in_channels = [256, 512, 1024, 2048]
    scales = [56, 28, 14, 7]  #
    inputs = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]
    # inputs[0].shape = torch.Size([1, 256, 56, 56])
    #


    outputs = fpn(inputs)
    for i in range(len(outputs)):
        print(f'outputs[{i}].shape = {outputs[i].shape}')

    # outputs[0].shape = torch.Size([1, 256, 28, 28])
    # outputs[1].shape = torch.Size([1, 256, 14, 14])
    # outputs[2].shape = torch.Size([1, 256, 7, 7])
    # outputs[3].shape = torch.Size([1, 256, 4, 4])
