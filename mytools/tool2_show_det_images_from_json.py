#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool2_merge_fewshot_json.py
# @Time      :2022/3/3 下午12:25
# @Author    :Yangliang
import os
import json
import cv2


def show_need_ann(gt_json_file, det_json_file, data_root, image_save_dir):
    start_image_id = 581900

    # image_save_dir = image_dir+"_show"
    if not os.path.exists(image_save_dir):
        os.mkdir(image_save_dir)
    with open(gt_json_file, 'r') as f:
        gt_json_info = json.load(f)

    with open(det_json_file, 'r') as f:
        det_json_info = json.load(f)

    images = gt_json_info['images']
    annotations = det_json_info
    images_label = {}
    for image in images:
        if image['id']<start_image_id:
            continue
        images_label[image['id']] = {'image_info': image, 'ann_info': []}
    # print(images_label[0])
    for ann in annotations:
        image_id = ann['image_id']
        if image_id<start_image_id:
            continue
        images_label[image_id]['ann_info'].append(ann)


    for image_id in images_label.keys():
        image_name = images_label[image_id]['image_info']['file_name']
        image_path = os.path.join(data_root, image_name)
        img = cv2.imread(image_path)
        for ann in images_label[image_id]['ann_info']:
            score = ann['score']
            if score<0.5:
                continue
            category_id = ann['category_id']
            x, y, w, h = [int(x) for x in ann['bbox']]
            # x1 = int(x-w/2)
            # x2 = int(x+w/2)
            # y1 = int(y-h/2)
            # y2 = int(y+h/2)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(category_id), (x, y), font, 1.2, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(image_save_dir, os.path.basename(image_name)), img)


if __name__ == "__main__":
    gt_json_file = "/media/E_4TB/YL/mmlab/mmfewshot/data/few_shot_ann/xyb_crop/annotations/val.json"
    det_json_file = "/media/E_4TB/YL/mmlab/mmfewshot/result/result.bbox.json"
    show_need_ann(gt_json_file, det_json_file, data_root='/media/E_4TB/YL/mmlab/mmfewshot/data', image_save_dir="/media/E_4TB/YL/mmlab/mmfewshot/result/det")

