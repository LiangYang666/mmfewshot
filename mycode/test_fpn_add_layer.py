#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :test_fpn_add_layer.py
# @Time      :2022/5/18 下午10:39
# @Author    :Yangliang
import torch

from mmfewshot.detection.models import build_shared_head
fpn_add_layer = dict(type='MetaRCNNFPNAddLayer')

if __name__ == "__main__":
    fpn_add_layer_model = build_shared_head(fpn_add_layer)

    input = torch.randn(1, 256, 14, 14)
    output = fpn_add_layer_model(input)
    print(output.shape)
