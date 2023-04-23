# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.utils import ConfigDict
from mmdet.models.builder import MODELS
from torch import Tensor

# AGGREGATORS are used for aggregate features from different data
# pipelines in meta-learning methods, such as attention rpn.
AGGREGATORS = MODELS


def build_aggregator(cfg: ConfigDict) -> nn.Module:
    """Build aggregator."""
    return AGGREGATORS.build(cfg)


@AGGREGATORS.register_module()
class AggregationLayer(BaseModule):
    """Aggregate query and support features with single or multiple aggregator.
    Each aggregator return aggregated results in different way.

    Args:
        aggregator_cfgs (list[ConfigDict]): List of fusion function.
        init_cfg (ConfigDict | None): Initialization config dict. Default: None
    """

    def __init__(self,
                 aggregator_cfgs: List[ConfigDict],
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.aggregator_list = nn.ModuleList()
        self.num_aggregators = len(aggregator_cfgs)
        aggregator_cfgs_ = copy.deepcopy(aggregator_cfgs)
        for cfg in aggregator_cfgs_:
            self.aggregator_list.append(build_aggregator(cfg))

    def forward(self, query_feat: Tensor,
                support_feat: Tensor) -> List[Tensor]:
        """Return aggregated features of query and support through single or
        multiple aggregators.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with
                shape (N, C, H, W).

        Returns:
            list[Tensor]: List of aggregated features.
        """
        out = []
        for i in range(self.num_aggregators):
            out.append(self.aggregator_list[i](query_feat, support_feat))
        return out


@AGGREGATORS.register_module()
class DenseRelationDistillAggregator(BaseModule):
    def __init__(self, in_channels=256, key_channels=32, val_channels=128, out_channels=256, dense_sum=True, init_cfg: Optional[ConfigDict] = None):
        super(DenseRelationDistillAggregator, self).__init__(init_cfg=init_cfg)
        # self.key_q = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        # self.value_q = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.key_t = nn.Conv2d(in_channels, key_channels, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.value_t = nn.Conv2d(in_channels, val_channels, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.sum = dense_sum
        if self.sum:    # q0 q1 q2 q3 q4 是查询集特增图的五个层级
            self.key_q0 = nn.Conv2d(in_channels, key_channels, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.value_q0 = nn.Conv2d(in_channels, val_channels, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.key_q1 = nn.Conv2d(in_channels, key_channels, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.value_q1 = nn.Conv2d(in_channels, val_channels, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.key_q2 = nn.Conv2d(in_channels, key_channels, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.value_q2 = nn.Conv2d(in_channels, val_channels, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.key_q3 = nn.Conv2d(in_channels, key_channels, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.value_q3 = nn.Conv2d(in_channels, val_channels, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.key_q4 = nn.Conv2d(in_channels, key_channels, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.value_q4 = nn.Conv2d(in_channels, val_channels, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.bnn0 = nn.BatchNorm2d(256)
            self.bnn1 = nn.BatchNorm2d(256)
            self.bnn2 = nn.BatchNorm2d(256)
            self.bnn3 = nn.BatchNorm2d(256)
            self.bnn4 = nn.BatchNorm2d(256)
            self.combine = nn.Conv2d(512, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, features, attentions):
        features = list(features)
        if isinstance(attentions, dict):
            for i in range(len(attentions)):
                if i == 0:
                    atten = attentions[i].unsqueeze(0)
                else:
                    atten = torch.cat((atten, attentions[i].unsqueeze(0)), dim=0)
            attentions = atten.cuda()
        output = []
        h, w = attentions.shape[2:]  # ??? ??? ??
        ncls = attentions.shape[0]  # ??? ???
        key_t = self.key_t(attentions)
        val_t = self.value_t(attentions)
        for idx in range(len(features)):  # ????
            feature = features[idx]
            bs = feature.shape[0]
            H, W = feature.shape[2:]
            feature = F.interpolate(feature, size=(h, w), mode='bilinear', align_corners=True)
            key_q = eval('self.key_q' + str(idx))(feature).view(bs, 32, -1)  # 4,32,256
            val_q = eval('self.value_q' + str(idx))(feature)  # 4,128,16,16
            for i in range(bs):  # ?batch??????????
                kq = key_q[i].unsqueeze(0).permute(0, 2, 1)
                vq = val_q[i].unsqueeze(0)

                p = torch.matmul(kq, key_t.view(ncls, 32, -1))
                p = F.softmax(p, dim=1)

                val_t_out = torch.matmul(val_t.view(ncls, 128, -1), p).view(ncls, 128, h, w)
                for j in range(ncls):
                    if (j == 0):
                        final_2 = torch.cat((vq, val_t_out[j].unsqueeze(0)), dim=1)
                    else:
                        final_2 += torch.cat((vq, val_t_out[j].unsqueeze(0)), dim=1)
                if (i == 0):
                    final_1 = final_2
                else:
                    final_1 = torch.cat((final_1, final_2), dim=0)
            final_1 = F.interpolate(final_1, size=(H, W), mode='bilinear', align_corners=True)
            if self.sum:
                final_1 = eval('self.bnn' + str(idx))(final_1)

            output.append(final_1)

        if self.sum:
            for i in range(len(output)):
                output[i] = self.combine(torch.cat((features[i], output[i]), dim=1))
        output = tuple(output)

        return output


@AGGREGATORS.register_module()
class DepthWiseCorrelationAggregator(BaseModule):
    """Depth-wise correlation aggregator.

    Args:
        in_channels (int): Number of input features channels.
        out_channels (int): Number of output features channels.
            Default: None.
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.
        init_cfg (ConfigDict | None): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 with_fc: bool = False,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert in_channels is not None, \
            "DepthWiseCorrelationAggregator require config of 'in_channels'."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_fc = with_fc
        if with_fc:
            assert out_channels is not None, 'out_channels is expected.'
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feat: Tensor, support_feat: Tensor) -> Tensor:
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (1, C, H, W).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """
        assert query_feat.size(1) == support_feat.size(1), \
            'mismatch channel number between query and support features.'

        feat = F.conv2d(
            query_feat,
            support_feat.permute(1, 0, 2, 3),
            groups=self.in_channels)
        if self.with_fc:
            assert feat.size(2) == 1 and feat.size(3) == 1, \
                'fc layer requires the features with shape (N, C, 1, 1)'
            feat = self.fc(feat.squeeze(3).squeeze(2))
            feat = self.norm(feat)
            feat = self.relu(feat)
        return feat


@AGGREGATORS.register_module()
class DifferenceAggregator(BaseModule):
    """Difference aggregator.

    Args:
        in_channels (int): Number of input features channels. Default: None.
        out_channels (int): Number of output features channels. Default: None.
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.
        init_cfg (ConfigDict | None): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 with_fc: bool = False,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.with_fc = with_fc
        self.in_channels = in_channels
        self.out_channels = out_channels
        if with_fc:
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feat: Tensor, support_feat: Tensor) -> Tensor:
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (N, C, H, W).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """
        assert query_feat.size(1) == support_feat.size(1), \
            'mismatch channel number between query and support features.'
        feat = query_feat - support_feat
        if self.with_fc:
            assert feat.size(2) == 1 and feat.size(3) == 1, \
                'fc layer requires the features with shape (N, C, 1, 1)'
            feat = self.fc(feat.squeeze(3).squeeze(2))
            feat = self.norm(feat)
            feat = self.relu(feat)
        return feat


@AGGREGATORS.register_module()
class DotProductAggregator(BaseModule):
    """Dot product aggregator.

    Args:
        in_channels (int): Number of input features channels. Default: None.
        out_channels (int): Number of output features channels. Default: None.
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.
        init_cfg (ConfigDict | None): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 with_fc: bool = False,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.with_fc = with_fc
        self.in_channels = in_channels
        self.out_channels = out_channels
        if with_fc:
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feat: Tensor, support_feat: Tensor) -> Tensor:
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (N, C, H, W).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """
        assert query_feat.size()[1:] == support_feat.size()[1:], \
            'mismatch channel number between query and support features.'
        feat = query_feat.mul(support_feat)
        if self.with_fc:
            assert feat.size(2) == 1 and feat.size(3) == 1, \
                'fc layer requires the features with shape (N, C, 1, 1)'
            feat = self.fc(feat.squeeze(3).squeeze(2))
            feat = self.norm(feat)
            feat = self.relu(feat)
        return feat
