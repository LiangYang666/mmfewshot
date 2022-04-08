# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import shutil
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmfewshot.detection.datasets import (build_dataloader, build_dataset,
                                          get_copy_dataset_type)
from mmfewshot.detection.models import build_detector


def check_create_dirs(dirs):
    if isinstance(dirs, str):
        dirs = [dirs]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print(f"\t--create dir*** {dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMFewShot test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('-input', help='directory where source images will be detected')
    parser.add_argument('-output', help='directory where painted images will be saved')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--save-support-heatmap', action='store_true', help='whether to save the support heat map')
    parser.add_argument(
        '--save-query-heatmap', action='store_true', help='whether to save the query heat map')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
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
    if args.output is not None:
        painted_dir = os.path.join(args.output, "painted_images")
        args.show_dir = painted_dir
        heatmap_dir = os.path.join(args.output, "heatmaps")
        txt_result_dir = os.path.join(args.output, "txt_result")
        check_create_dirs([painted_dir, heatmap_dir, txt_result_dir])
        # if args.cfg_options is None:
        #     args.cfg_options = dict()
        # if "evaluation.jsonfile_prefix" not in args.cfg_options.keys():
        #     args.cfg_options['evaluation.jsonfile_prefix'] = os.path.join(args.output, "result")
        if args.eval_options is None:
            args.eval_options = dict()
        if "jsonfile_prefix" not in args.eval_options.keys():
            args.eval_options['jsonfile_prefix'] = os.path.join(args.output, "result")
        if "iou_thrs" not in args.eval_options.keys():
            args.eval_options['iou_thrs'] = [0.5]

        if args.out is None:
            args.out = os.path.join(args.output, "result.pkl")
        if args.eval is None:
            args.eval = ["bbox"]

    assert args.out or args.eval or args.show \
        or args.show_dir, (
            'Please specify at least one operation (save/eval/show the '
            'results / save the results) with the argument "--out", "--eval"',
            '"--show" or "--show-dir"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.output is not None:
        cfg.heatmap_dir = heatmap_dir
        cfg.save_support_heatmap = args.save_support_heatmap
        cfg.save_query_heatmap = args.save_query_heatmap

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

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # pop frozen_parameters
    cfg.model.pop('frozen_parameters', None)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model)
    model.cfg = cfg

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

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

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        show_kwargs = dict(show_score_thr=args.show_score_thr)
        if cfg.data.get('model_init', None) is not None:
            from mmfewshot.detection.apis import (single_gpu_model_init,
                                                  single_gpu_test)
            single_gpu_model_init(model, model_init_dataloader)
        else:
            from mmdet.apis.test import single_gpu_test
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  **show_kwargs)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        if cfg.data.get('model_init', None) is not None:
            from mmfewshot.detection.apis import (multi_gpu_model_init,
                                                  multi_gpu_test)
            multi_gpu_model_init(model, model_init_dataloader)
        else:
            from mmdet.apis.test import multi_gpu_test
        outputs = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
        )

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            if args.output is not None:
                result_files, tmp_dir = dataset.format_results(outputs, eval_kwargs["jsonfile_prefix"])
            else:
                result = dataset.evaluate(outputs, **eval_kwargs)
                print(result)

    if args.output is not None:
        def write_to_result_txt(file, result, categories, save_dir, score_thr=0.3):
            txt_file = os.path.join(save_dir, file.rsplit('.', 1)[0] + '.txt')
            dir = os.path.dirname(txt_file)
            if not os.path.exists(dir):
                os.makedirs(dir)
            assert len(result) == len(categories)
            with open(txt_file, 'w') as f:
                for i, category_result in enumerate(result):
                    category_result = category_result.tolist()
                    for category_bbox_result in category_result:
                        if category_bbox_result[-1] >= score_thr:
                            category_bbox_result = [str(round(x, 3)) for x in category_bbox_result]
                            f.write(f"{categories[i]}({i}) " + " ".join(category_bbox_result).strip() + "\n")
        assert len(outputs) == len(data_loader)
        for i, data in enumerate(data_loader):
            img_ori_filename = data["img_metas"][0].data[0][0]['ori_filename']
            write_to_result_txt(img_ori_filename, outputs[i], model.module.CLASSES, txt_result_dir, args.show_score_thr)
        map_tmp_dir = os.path.join(args.output, "map_dir")
        if os.path.exists(map_tmp_dir):
            shutil.rmtree(map_tmp_dir)
        true_txt_dir = os.path.join(map_tmp_dir, 'ground-truth')
        dect_txt_dir = os.path.join(map_tmp_dir, 'detection-results')
        check_create_dirs([map_tmp_dir, true_txt_dir, dect_txt_dir])
        cocoGt = data_loader.dataset.coco
        predictions = mmcv.load(args.eval_options['jsonfile_prefix']+'.bbox.json')
        cocoDt = cocoGt.loadRes(predictions)
        ids = cocoGt.get_img_ids()
        for id in ids:
            # imgGt = cocoGt.loadImgs(id)[0]
            imgDt = cocoDt.loadImgs(id)[0]
            file_name = os.path.basename(imgDt['file_name'].rsplit(".", 1)[0])+'.txt'
            with open(os.path.join(true_txt_dir, file_name), 'w') as f:
                for bbox, label in zip(imgDt['ann']['bboxes'], imgDt['ann']['labels']):
                    category_name = cocoDt.dataset['categories'][label]['name']
                    bbox = [str(x) for x in bbox]
                    f.write((category_name+" "+" ".join(bbox)).strip()+"\n")
            with open(os.path.join(dect_txt_dir, file_name), 'w') as f:
                dect_info = cocoDt.imgToAnns[id]
                for each in dect_info:
                    assert id == each['image_id']
                    label = each['category_id']
                    bb = each['bbox']
                    bbox = [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]]
                    bbox = [str(x) for x in bbox]
                    category_name = cocoDt.loadCats(int(label))[0]['name']
                    score = str(each['score'])
                    f.write((category_name+" "+score+" "+" ".join(bbox)).strip() + "\n")

        from mytools.map import cac_map
        total, correct = cac_map(os.path.abspath(map_tmp_dir))
        data = []
        headers = ["category", "correct", "error", "accuracy", "false_alarm"]
        data.append(headers)
        for name in sorted(total):
            accuracy = correct[name]*1.0/total[name]
            temp = [name, correct[name], total[name]-correct[name],
                    str(round(accuracy*100, 1))+"%", str(round((1-accuracy)*100, 1))+"%"]
            data.append(temp)

        from terminaltables import AsciiTable
        table = AsciiTable(data)
        print(table.table)
        # with open(args.output+"/result.txt", 'w') as f:
        #     f.write(str(table.table))
        for i in range(len(data)):
            data[i] = [str(x) for x in data[i]]
            data[i] = '\t'.join(data[i])
        with open(args.output+"/result.txt", 'w') as f:
            f.write('\n'.join(data))


if __name__ == '__main__':
    main()
