#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool1_generate_val_json.py
# @Time      :2022/3/2 下午11:55
# @Author    :Yangliang
import json
import os


def check_and_makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"make dir {dir}")


def save_src_val_to_val_json_file(src_json_file, dst_json_file, data_prefix):
    save_dir = os.path.dirname(dst_json_file)
    check_and_makedirs(save_dir)
    with open(src_json_file, 'r') as f:
        src_json = json.load(f)
    for image_info in src_json['images']:       # 将图片信息中的 id增加，并在file_name上添加prefix
        image_info['file_name'] = data_prefix + image_info['file_name']
    with open(dst_json_file, 'w') as f:
        json.dump(src_json, f, indent=2)


def save_src_train_to_fewshot_json_dir(src_json_file, dst_json_dir, data_prefix):
    check_and_makedirs(dst_json_dir)
    with open(src_json_file, 'r') as f:
        src_json = json.load(f)
    for image_info in src_json['images']:       # 将图片信息中的 id增加，并在file_name上添加prefix
        image_info['file_name'] = data_prefix + image_info['file_name']

    novel_each_json = {}
    new_images_id_info = {}

    for image in src_json['images']:
        new_images_id_info[image['id']] = image

    for category in src_json['categories']:
        category_name = category['name']
        category_id = category['id']
        novel_each_json[category_name] = {'categories': src_json['categories'], 'annotations': [], 'images': []}
        add_images_ids = set()
        for ann in src_json['annotations']:
            if category_id == ann['category_id']:
                add_images_ids.add(ann['image_id'])
                novel_each_json[category_name]['annotations'].append(ann)
        for image_id in add_images_ids:
            novel_each_json[category_name]['images'].append(new_images_id_info[image_id])
        with open(os.path.join(dst_json_dir, f'full_box_10shot_{category_name}_trainval.json'), 'w') as f:
            json.dump(novel_each_json[category_name], f, indent=2)


if __name__ == "__main__":
    print("start merge val json------------***********************")
    src_train_json_file = "/media/E_4TB/YL/mmlab/mmfewshot/data/xyb/annotations/train.json"
    src_val_json_file = "/media/E_4TB/YL/mmlab/mmfewshot/data/xyb/annotations/val.json"

    dst_val_json_file = "/media/E_4TB/YL/mmlab/mmfewshot/data/few_shot_ann/xyb/annotations/val.json"
    dst_few_shot_json_dir = "/media/E_4TB/YL/mmlab/mmfewshot/data/few_shot_ann/xyb/benchmark_10shot"

    val_data_prefix = 'xyb/images/val/'
    train_data_prefix = 'xyb/images/train/'

    save_src_train_to_fewshot_json_dir(src_train_json_file, dst_few_shot_json_dir, train_data_prefix)
    save_src_val_to_val_json_file(src_val_json_file, dst_val_json_file, val_data_prefix)
