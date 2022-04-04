#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool6_labelme2coco.py
# @Time      :2022/4/3 下午6:57
# @Author    :Yangliang
import json
import os
import shutil

import cv2
from pycocotools.coco import COCO
from terminaltables import AsciiTable
from tqdm import tqdm

img_start_id = 3000
ann_start_id = 10000

categories_info = {'Tank': {'id': 1, 'total': 0},     # 如果类别指定不正确 将会报错 由于程序先将所有类别打印出来，因此可以查看输出
                   'Truck': {'id': 2, 'total': 0},     # id为将保存到coco中的类别id (可自己指定)，total是作为统计使用的，初始为0即可
                   'Car': {'id': 3, 'total': 0},
                   'Tent': {'id': 4, 'total': 0}}

if __name__ == "__main__":
    data_labelme_dir = '/media/E_4TB/YL/mmlab/mmfewshot/mytools/data/可见光/3军用设施裁剪+删原图'  # labelme 路径
    dst_img_dir = '/media/E_4TB/YL/mmlab/mmfewshot/mytools/data/可见光/images'  # 保存图片的目标文件夹
    dst_json_file = '/media/E_4TB/YL/mmlab/mmfewshot/mytools/data/可见光/all.json'  # 保存生成coco json文件
    if not os.path.exists(dst_img_dir):
        os.mkdir(dst_img_dir)
    img_id = img_start_id - 1
    ann_id = ann_start_id - 1
    files = os.listdir(data_labelme_dir)
    imgs_suffix = ['jpg', 'jpeg', 'tif', 'png']
    categories = [{'id': categories_info[x]['id'], 'name': x} for x in categories_info.keys()]
    dst_json = {'categories': categories, 'images': [], 'annotations': []}
    imgs = [x for x in files if x.rsplit('.')[-1].lower() in imgs_suffix]
    categories_get = []
    for img_name in tqdm(imgs):
        name = img_name.rsplit('.')[0]
        labelme_json_file = name + '.json'
        if labelme_json_file not in files: continue
        with open(os.path.join(data_labelme_dir, labelme_json_file), 'r') as f:
            labelme_json = json.load(f)
        for shape in labelme_json['shapes']:
            label = shape['label']
            if label not in categories_get:
                categories_get.append(label)
    print(f'all categories get: {sorted(categories_get)}')

    for img_name in tqdm(imgs):
        name = img_name.rsplit('.')[0]
        labelme_json_file = name + '.json'
        if labelme_json_file not in files: continue
        img = cv2.imread(os.path.join(data_labelme_dir, img_name))
        img_id += 1
        img_info = {"height": img.shape[0], "width": img.shape[1],
                    "id": img_id, "file_name": img_name}
        dst_json['images'].append(img_info)
        with open(os.path.join(data_labelme_dir, labelme_json_file), 'r') as f:
            labelme_json = json.load(f)
        for shape in labelme_json['shapes']:
            label = shape['label']
            assert label in categories_info, label + ' is not a valid class'
            category_id = categories_info[label]['id']
            categories_info[label]['total'] += 1
            points = shape['points']
            xs = sorted([points[0][0], points[1][0]])
            ys = sorted([points[0][1], points[1][1]])
            x1, x2 = xs
            y1, y2 = ys
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            area = bbox[2] * bbox[3]
            ann_id += 1
            ann_info = {'id': ann_id, 'image_id': img_id,
                        'category_id': category_id, 'bbox': bbox, 'area': area}
            dst_json['annotations'].append(ann_info)
            shutil.copy(os.path.join(data_labelme_dir, img_name), os.path.join(dst_img_dir, img_name))

    with open(dst_json_file, 'w') as f:
        json.dump(dst_json, f, indent=2)

    table = [['id', 'name', 'total']]
    for name in categories_info.keys():
        table.append([categories_info[name]['id'], name, categories_info[name]['total']])
    table = AsciiTable(table)
    print(table.table)
