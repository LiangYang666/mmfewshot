#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool6_lebelme2fixed_scale.py
# @Time      :2022/4/8 上午10:50
# @Author    :Yangliang
import json
import os
import cv2
from tqdm import tqdm

'''
将labelme打的标签和图片缩放到固定宽度，高度自适应变化
'''

fixed_width = 1080  # 设置宽度目标大小
img_types = ['jpg', 'jpeg', 'tif', 'png']

if __name__ == "__main__":
    src_labelme_dir = "/media/E_4TB/YL/mmlab/mmfewshot/mytools/data/红外/2挑选裁剪-删除原图"      # 原labelme路径
    dst_labelme_dir = "/media/E_4TB/YL/mmlab/mmfewshot/mytools/data/红外/2挑选裁剪-删除原图-scale"    # 新labelme路径
    if not os.path.exists(dst_labelme_dir):
        os.mkdir(dst_labelme_dir)
    src_files = os.listdir(src_labelme_dir)
    src_json_files = [x for x in src_files if x.endswith('.json')]
    img_files = [x for x in src_files if x.rsplit('.', 1)[0]+'.json' in src_json_files and x.rsplit('.', 1)[-1].lower() in img_types]
    for img_file in tqdm(img_files):
        img_path = os.path.join(src_labelme_dir, img_file)
        json_path = os.path.join(src_labelme_dir, img_file.rsplit('.', 1)[0]+'.json')
        img = cv2.imread(img_path)
        scale_factor = fixed_width / img.shape[1]
        img = cv2.resize(img, dsize=None, fx=scale_factor, fy=scale_factor)
        cv2.imwrite(os.path.join(dst_labelme_dir, img_file), img)
        with open(json_path, 'r') as f:
            json_info = json.load(f)
        json_info['imageData'] = None
        json_info["imageHeight"] = img.shape[0]
        json_info["imageWidth"] = img.shape[1]
        for shape in json_info['shapes']:
            for point in shape['points']:
                point[0] *= scale_factor
                point[1] *= scale_factor
        with open(os.path.join(dst_labelme_dir, img_file.rsplit('.', 1)[0]+'.json'), 'w') as f:
            json.dump(json_info, f, indent=2)
