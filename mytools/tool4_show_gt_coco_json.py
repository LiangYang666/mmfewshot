#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool4_show_gt_coco_json.py
# @Time      :2022/3/27 下午2:04
# @Author    :Yangliang
import os

import cv2
from tool3_data_augmentation import get_json_info
from tqdm import tqdm

'''
通过coco json文件
在图片上显示真实标签框
'''

if __name__ == "__main__":
    gt_json_file = "/media/E_4TB/YL/mmlab/mmfewshot/mytools/data/红外/2挑选裁剪-删除原图-scale-coco/all.json" # 标签json文件
    src_image_dir = '/media/E_4TB/YL/mmlab/mmfewshot/mytools/data/红外/2挑选裁剪-删除原图-scale-coco/images'                           # 图片路径
    plot_image_dir = '/media/E_4TB/YL/mmlab/mmfewshot/mytools/data/红外/2挑选裁剪-删除原图-scale-coco/images_display'                  # 绘制了结果框的图片保存路径
    if not os.path.exists(plot_image_dir):
        os.mkdir(plot_image_dir)

    image_info, origin_json = get_json_info(gt_json_file)
    categories = origin_json['categories']
    categories = {x['id']:x['name'] for x in categories}
    names = sorted(image_info.keys())
    for name in tqdm(names):
        annotations = image_info[name]['annotations']
        name = os.path.basename(name)
        image_path = os.path.join(src_image_dir, name)
        img = cv2.imread(image_path)
        for ann in annotations:
            category_id = ann['category_id']
            x, y, w, h = [int(x) for x in ann['bbox']]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, categories[category_id], (x, y), font, 1.2, (255, 255, 255), 2)

        cv2.imwrite(os.path.join(plot_image_dir, name), img)

