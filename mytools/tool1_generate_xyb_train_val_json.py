#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool1_generate_val_json.py
# @Time      :2022/3/2 下午11:55
# @Author    :Yangliang
import argparse
import json
import os
import warnings

'''
    从train.json val.json 生成小样本数据集标签
    1.生成n个小样本训练json
    2.生成1个小样本验证json
    3.修改配置文件中的类别数量
    4.修改数据集定义中的类别信息
'''


def check_and_makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"make dir {dir}")


def save_src_val_to_val_json_file(src_json_file, dst_json_file, data_prefix):
    if src_json_file is not None and os.path.exists(src_json_file):
        save_dir = os.path.dirname(dst_json_file)
        check_and_makedirs(save_dir)
        with open(src_json_file, 'r') as f:
            src_json = json.load(f)
        for image_info in src_json['images']:       # 将图片信息中的 id增加，并在file_name上添加prefix
            image_info['file_name'] = data_prefix + image_info['file_name']
        with open(dst_json_file, 'w') as f:
            print(f"\t----***----save fewshot val json to  {dst_val_json_file}")
            json.dump(src_json, f, indent=2)
    else:
        warnings.warn("There is no validate json file")


def save_src_train_to_fewshot_json_dir(src_json_file, dst_json_dir, data_prefix):
    assert os.path.exists(src_json_file), 'There is no train json file'
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
        file_path = os.path.join(dst_json_dir, f'full_box_10shot_{category_name}_trainval.json')
        with open(file_path, 'w') as f:
            print(f"\t-1---***----save fewshot train jsons to  {file_path}")
            json.dump(novel_each_json[category_name], f, indent=2)

    print("\t----***----")
    categories_dic = {}
    for category in src_json['categories']:
        category_name = category['name']
        category_id = category['id']
        categories_dic[category_id] = category_name
        print(f"\tid:{category_id} \tname: {category_name}")
    category_names = []
    for category_id in sorted(categories_dic.keys()):
        category_names.append(categories_dic[category_id])
    print(f"\tall_total {len(category_names)} {category_names}")
    return category_names


def change_config_file(category_names,
                       config_file="mytools/xyb-rcnn_r50_c4_8xb4_novel-fine-tuning.py",
                       dataset_file="mmfewshot/detection/datasets/xyb.py"):

    change_flag = False
    with open(config_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("num_classes"):
                infos = line.split("=")
                infos[0] = infos[0].strip()
                infos[1] = f"{len(category_names)}\n"
                lines[i] = ' = '.join(infos)
                change_flag = True
                break
    if change_flag:
        print(f"\t\t\t{config_file}")
        with open(config_file, 'w') as f:
            for line in lines:
                f.write(line)

    change_flag = False
    with open(dataset_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("COCO_SPLIT"):
                assert "NOVEL_CLASSES" in lines[i+1]
                if "NOVEL_CLASSES" in lines[i+1]:
                    infos = lines[i+1].split("=")
                    infos[1] = str(tuple(category_names))+"\n"
                    lines[i+1] = "=".join(infos)
                    change_flag = True
                    break

    if change_flag:
        print(f"\t\t\t{dataset_file}\n")
        with open(dataset_file, 'w') as f:
            for line in lines:
                f.write(line)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a FewShot model')
    parser.add_argument('-train', '--train-json',
                        default='data/xyb/annotations/train.json',
                        help='train json file path')
    parser.add_argument('-val', '--val-json',
                        default='data/xyb/annotations/val.json',
                        help='validate json file path')

    parser.add_argument('-config-file',
                        default="mytools/xyb-rcnn_r50_c4_8xb4_novel-fine-tuning.py",
                        help='config json file path')

    parser.add_argument('-dataset-file',
                        default="mmfewshot/detection/datasets/xyb.py",
                        help='dataset file path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    src_train_json_file = args.train_json
    src_val_json_file = args.val_json

    dst_val_json_file = "data/few_shot_ann/xyb/annotations/val.json"
    dst_few_shot_json_dir = "data/few_shot_ann/xyb/benchmark_10shot"

    val_data_prefix = 'xyb/images/val/'
    train_data_prefix = 'xyb/images/train/'

    category_names = save_src_train_to_fewshot_json_dir(src_train_json_file, dst_few_shot_json_dir, train_data_prefix)
    save_src_val_to_val_json_file(src_val_json_file, dst_val_json_file, val_data_prefix)

    config_file = "mytools/xyb-rcnn_r50_c4_8xb4_novel-fine-tuning.py"
    dataset_file = "mmfewshot/detection/datasets/xyb.py"

    print(f"\t----***----change classes information  ")
    change_config_file(category_names, config_file, dataset_file)
