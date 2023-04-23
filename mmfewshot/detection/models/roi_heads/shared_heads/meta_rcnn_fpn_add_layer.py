# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import Conv2d, build_conv_layer
from mmcv.runner import BaseModule
from mmdet.models.builder import SHARED_HEADS
from torch import Tensor


@SHARED_HEADS.register_module(force=True)
class MetaRCNNFPNAddLayer(BaseModule):
    """Shared addLayer for metarcnn-fpn .

    It provides different forward logics for query and support images.
    """

    def __init__(self, init_cfg=None, pretrained=None):
        super().__init__(init_cfg)
        self.max_pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()
        self.conv_cfg = None
        self.share_add_layer = build_conv_layer(
            self.conv_cfg,  # from config of ResNet
            256,
            2048,
            kernel_size=7,
            stride=1,
            padding=2,
            bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for query images.

        Args:
            x (Tensor): Features from backbone with shape (N, C, H, W).

        Returns:
            Tensor: Shape of (N, C).
        """
        out = self.share_add_layer(x)
        out = out.max(3).values.max(2).values
        return out

    def forward_support(self, x: Tensor) -> Tensor:
        """Forward function for support images.

        Args:
            x (Tensor): Features from backbone with shape (N, C, H, W).

        Returns:
            Tensor: Shape of (N, C).
        """
        x = self.max_pool(x)
        out = self.share_add_layer(x)
        out = self.sigmoid(out)
        out = out.max(3).values.max(2).values
        return out
