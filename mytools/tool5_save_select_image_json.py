#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool5_save_select_image_json.py
# @Time      :2022/3/27 下午9:31
# @Author    :Yangliang
import json
import os
import shutil
'''
根据select_img_dir剩下的图像
保留扩充图中需要留下的图像，并生成新的json文件
'''
if __name__ == "__main__":
    select_img_dir = '/media/E_4TB/YL/mmlab/mmfewshot/result_show_aug/select'       # 挑选出来的图片 但是是绘制了预测框的
    src_aug_img_dir = '/media/E_4TB/YL/mmlab/mmfewshot/data/xyb_aug/images/val_aug'     # 原始数据扩充后的图片
    save_select_origin_img_dir = '/media/E_4TB/YL/mmlab/mmfewshot/data/xyb/images/val_aug'  # 扩充图片中 根据挑选图 保存原始未预测框的的路径
    src_json_file = '/media/E_4TB/YL/mmlab/mmfewshot/data/xyb_aug/annotations/val_aug.json'  # 原始数据扩充后的标签
    dst_json_file = '/media/E_4TB/YL/mmlab/mmfewshot/data/few_shot_ann/xyb/annotations/val_select.json'   # 挑选后的标签
    with open(src_json_file, 'r') as f:
        src_json = json.load(f)
    dst_json = {'images': [], 'categories': src_json['categories'], 'annotations': []}
    files = os.listdir(select_img_dir)
    need_save_image_ids = []
    for image in src_json['images']:
        if os.path.basename(image['file_name']) in files:
            id = image['id']
            need_save_image_ids.append(id)
            dst_json['images'].append(image)
    print(f"\t all images {len(need_save_image_ids)}")
    for ann in src_json['annotations']:
        if ann['image_id'] in need_save_image_ids:
            dst_json['annotations'].append(ann)
    print(f"\t all anns {len(dst_json['annotations'])}")
    with open(dst_json_file, 'w') as f:
        json.dump(dst_json, f, indent=2)
    if not os.path.exists(save_select_origin_img_dir):
        shutil.rmtree(save_select_origin_img_dir)
        os.mkdir(save_select_origin_img_dir)
    for file in files:
        src = os.path.join(src_aug_img_dir, file)
        dst = os.path.join(save_select_origin_img_dir, file)
        shutil.copyfile(src, dst)




