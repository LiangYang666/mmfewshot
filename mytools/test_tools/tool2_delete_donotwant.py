#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool2_delete_dontwant.py
# @Time      :2022/3/17 下午7:52
# @Author    :Yangliang
import copy
import json
import os.path
import shutil
import datetime

will_delete_dir = '/media/E_4TB/YL/mmlab/mmfewshot/result_show_inf/painted_images/xyb_inf/images/del'        # 将要删除图片集路径
val_json_file = '/media/E_4TB/YL/mmlab/mmfewshot/data/few_shot_ann/xyb_inf/annotations/val.json'        # coco标签文件
val_dir = '/media/E_4TB/YL/mmlab/mmfewshot/data/xyb_inf/images/val'         # 图片文件夹

will_delete_imgs = os.listdir(will_delete_dir)
if __name__ == "__main__":
    time_str = datetime.datetime.now().strftime('-%m-%d_%H-%M')
    shutil.copy(val_json_file, val_json_file.rsplit('.')[0]+time_str+'.json')   # 备份json
    shutil.move(val_dir, val_dir+time_str)      # 备份图片文件夹
    os.mkdir(val_dir)
    with open(val_json_file, 'r') as f:
        val_json = json.load(f)
    delete_img_ids = []
    new_json = copy.deepcopy(val_json)
    new_json['images'] = []
    new_json['annotations'] = []
    for image in val_json["images"]:
        name = os.path.basename(image['file_name'])
        if name in will_delete_imgs:
            delete_img_ids.append(image['id'])
        else:
            shutil.copy(os.path.join(val_dir+time_str, name), os.path.join(val_dir, name))
            new_json['images'].append(image)
    for ann in val_json["annotations"]:
        if ann['image_id'] not in delete_img_ids:
            new_json['annotations'].append(ann)
    with open(val_json_file, 'w') as f:
        json.dump(new_json, f, indent=2)







