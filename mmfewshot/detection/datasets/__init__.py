# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseFewShotDataset
from .builder import build_dataloader, build_dataset
from .coco import FewShotCocoDataset  # COCO_SPLIT,
from .dataloader_wrappers import NWayKShotDataloader
from .dataset_wrappers import NWayKShotDataset, QueryAwareDataset
from .pipelines import CropResizeInstance, GenerateMask
from .utils import NumpyEncoder, get_copy_dataset_type
from .voc import VOC_SPLIT, FewShotVOCDataset
from .xyb import FewShotXYBDataset

__all__ = [
    'build_dataloader', 'build_dataset', 'QueryAwareDataset',
    'NWayKShotDataset', 'NWayKShotDataloader', 'BaseFewShotDataset',
    'FewShotVOCDataset', 'FewShotCocoDataset', 'FewShotXYBDataset',
'CropResizeInstance',
    'GenerateMask', 'NumpyEncoder',#  'COCO_SPLIT', 'VOC_SPLIT',
    'get_copy_dataset_type'
]
