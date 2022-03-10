# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.ops import RoIPool
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.parallel import collate, scatter
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmcv.image import tensor2imgs
from mmcv.parallel import is_module_wrapper
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

from torch import nn

from mmfewshot.detection.datasets import (build_dataloader, build_dataset,
                                          get_copy_dataset_type)
from mmfewshot.detection.models import build_detector, QuerySupportDetector

from mmfewshot.detection.apis import (inference_detector, init_detector,
                                      process_support_images)

from mmdet.apis import show_result_pyplot

from typing import List, Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMFewShot test (and eval) a model')
    parser.add_argument('-input', help='directory where source images will be detected')
    parser.add_argument('-output', help='directory where painted images will be saved')
    parser.add_argument('--config', default='./configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_xyb_10shot_novel-fine-tuning.py',help='test config file path')
    parser.add_argument('--checkpoint', default='./work_dirs/meta-rcnn_r50_c4_8xb4_xyb_10shot_novel-fine-tuning/iter_30000.pth', help='checkpoint file')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.1,
        help='score threshold (default: 0.3)')

    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--eval',
        type=str,
        default=['bbox'],
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
        args.cfg_options = args.options
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # currently only support single images testing
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    assert samples_per_gpu == 1, 'currently only support single images testing'


    # pop frozen_parameters
    cfg.model.pop('frozen_parameters', None)
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']

    # for meta-learning methods which require support template dataset
    # for model initialization.
    if cfg.data.get('model_init', None) is not None:
        cfg.data.model_init.pop('copy_from_train_dataset')
        model_init_samples_per_gpu = cfg.data.model_init.pop(
            'samples_per_gpu', 1)
        model_init_workers_per_gpu = cfg.data.model_init.pop(
            'workers_per_gpu', 1)
        if cfg.data.model_init.get('ann_cfg', None) is None:
            assert checkpoint['meta'].get('model_init_ann_cfg',
                                          None) is not None
            cfg.data.model_init.type = \
                get_copy_dataset_type(cfg.data.model_init.type)
            cfg.data.model_init.ann_cfg = \
                checkpoint['meta']['model_init_ann_cfg']
        model_init_dataset = build_dataset(cfg.data.model_init)
        # disable dist to make all rank get same data
        model_init_dataloader = build_dataloader(
            model_init_dataset,
            samples_per_gpu=model_init_samples_per_gpu,
            workers_per_gpu=model_init_workers_per_gpu,
            dist=False,
            shuffle=False)

    model = MMDataParallel(model, device_ids=[0])
    if cfg.data.get('model_init', None) is not None:
        from mmfewshot.detection.apis import (single_gpu_model_init,
                                              single_gpu_test)
        single_gpu_model_init(model, model_init_dataloader)
    else:
        from mmdet.apis.test import single_gpu_test
    model = model.module
    model.cfg = cfg
    files = os.listdir(args.input)

    prog_bar = mmcv.ProgressBar(len(files))
    for file in files:
        img = os.path.join(args.input, file)
        result = inference_detector(model, img)
        model.show_result(img, result, score_thr=args.show_score_thr, out_file=os.path.join(args.output, file))
        prog_bar.update(1)


if __name__ == '__main__':
    main()
