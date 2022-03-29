#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool3_data_augmentation.py
# @Time      :2022/3/24 下午8:52
# @Author    :Yangliang

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool2DataAugmentation.py
# @Time      :2021/8/3 下午5:16
# @Author    :Yangliang
import json
import os
import shutil
from tkinter import Tk

import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from matplotlib import pyplot as plt
from tqdm import tqdm
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from pycocotools.coco import COCO

# 进行数据扩充，增强策略为imgaug_seq
#

ia.seed(1)

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
my_resize = iaa.Resize({"height": "keep-aspect-ratio", "width": 1080})
imgaug_first = iaa.Sequential([my_resize])
imgaug_seq = iaa.Sequential(    # 策略
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.3),  # horizontally flip 50% of all images
        iaa.Flipud(0.2),  # vertically flip 20% of all images
        # crop images by -2.5% to 5% of their height/width
        # sometimes(iaa.CropAndPad(
        #     percent=(-0.03, 0.08),
        #     pad_mode=ia.ALL,
        #     pad_cval=(0, 255)
        # )),
        iaa.Sometimes(0.7, iaa.Affine(
            scale=(1.1,1.5),
            # scale images to 80-120% of their size, individually per axis
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
            rotate=(-60, 60),  # rotate by -45 to +45 degrees
            shear=(-30, 30),  # shear by -16 to +16 degrees
            # order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        iaa.Sometimes(0.8, iaa.AddToHueAndSaturation((-50, 50))),  # change hue and saturation
        # either change the brightness of the whole image (sometimes
        # per channel) or change the brightness of subareas

        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((3, 6),
                   [
                       # sometimes(iaa.Superpixels(p_replace=(0, 0.2), n_segments=100)),    # 去除临近填充
                       # convert images into their superpixel representation
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 2.0)),  # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(2, 4)),
                           # blur image using local means with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(3, 5)),
                           # blur image using local medians with kernel sizes between 2 and 7
                       ]),

                       # iaa.GaussianBlur((0, 2.0)),  # blur images with a sigma between 0 and 3.0
                       iaa.Sharpen(alpha=(0, 0.2), lightness=(0.85, 1.2)),  # sharpen images
                       iaa.Emboss(alpha=(0, 0.1), strength=(0, 1.5)),  # emboss images
                       # search either for all edges or for directed edges,
                       # blend the result with the original image using a blobby mask
                       iaa.SimplexNoiseAlpha(iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0.1, 0.16)),
                           iaa.DirectedEdgeDetect(alpha=(0.1, 0.16), direction=(0.0, 1.0)),
                       ])),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.015 * 255), per_channel=0.2),
                       # add gaussian noise to images

                       # iaa.Sometimes(0.3, iaa.Cartoon()),
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.02), per_channel=0.2),  # randomly remove up to 10% of the pixels
                           iaa.CoarseDropout((0.01, 0.02), size_percent=(0.01, 0.03), per_channel=0.2),
                       ]),
                       # iaa.Invert(0.05, per_channel=True),  # invert color channels
                       iaa.Add((-10, 10), per_channel=0.5),
                       # change brightness of images (by -10 to 10 of original value)
                       iaa.Pepper(0.03),  # 添加椒盐噪声

                       iaa.OneOf([
                           iaa.Multiply((0.9, 1.1), per_channel=0.5),
                           iaa.FrequencyNoiseAlpha(
                               exponent=(-1, 0),
                               first=iaa.Multiply((0.8, 1.2), per_channel=True),
                               second=iaa.LinearContrast((0.8, 1.3))
                           )
                       ]),
                       iaa.LinearContrast((0.8, 1.2), per_channel=0.5),  # improve or worsen the contrast
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       sometimes(iaa.ElasticTransformation(alpha=(0.9, 1.2), sigma=0.25)),
                       # move pixels locally around (with random strengths)
                       sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.02))),
                       # sometimes move parts of the image around
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.05))),

                       # iaa.ChangeColorTemperature((1100, 1500))
                   ],
                   random_order=True
                   ),
        my_resize
    ],
    random_order=True
)




def aug_save_img(src_images_dir, aug_img_dir, aug_times = 100, image_info=None, new_json=None, new_image_pre_dir=None):
    '''
    src_images_dir 原图像的保存路径
    aug_img_dir 增强后图片的保存路径
    aug_times 增强次数 即单张图像将扩充为的张数

    image_info 可选 如果有则为coco标签按照图片名保存的字典，
    new_json 可选 通过数据增强后新生成的coco标签文件（因为框位置将变化）

    new_image_pre_dir 可选 如果指定，则在新标签中的 file_name 将会加上其
    '''
    files = sorted(os.listdir(src_images_dir))
    img_suffixs = ['jpg', 'jpeg', 'png', 'tif']
    files = [x for x in files if x.rsplit('.', 1)[-1].lower() in img_suffixs]
    assert (image_info is None and new_json is None) or (image_info is not None and new_json is not None), "should not be only one None"
    assert new_json is None or 'categories' in new_json.keys()
    if new_json is not None:
        new_image_index = 0
        new_annotation_index = 0
    for file in tqdm(files):
        # src_img = plt.imread(os.path.join(src_images_dir, file))
        src_img = cv2.imread(os.path.join(src_images_dir, file), cv2.IMREAD_COLOR+cv2.IMREAD_IGNORE_ORIENTATION)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        suffix = file.rsplit('.', 1)[-1]
        bbs = None
        if image_info is not None:
            if file not in image_info.keys():
                continue
            annotations = image_info[file]['annotations']
            bboxes = []
            for annotation in annotations:
                bbox = annotation['bbox'].copy()
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                bbox.append(annotation['category_id'])
                bboxes.append(BoundingBox(*bbox))
            bbs = BoundingBoxesOnImage(bboxes, shape=src_img.shape)
        for t in range(aug_times):
            index = str(t+1).zfill(3)
            file_path = os.path.join(aug_img_dir, file.rsplit('.', 1)[0] + f'_{index}.' + suffix)
            if t == 0:
                temp_aug_func = imgaug_first
            else:
                temp_aug_func = imgaug_seq
            try:
                aug_img, after_bbs = temp_aug_func(image=src_img, bounding_boxes=bbs)
            except:
                print(f"错误: {file_path}")
            else:
                plt.imsave(file_path, aug_img)
                if new_json is not None:
                    json_image_file_name = os.path.basename(file_path)
                    if new_image_pre_dir is not None:
                        json_image_file_name = new_image_pre_dir + json_image_file_name
                    new_json['images'].append({'height': aug_img.shape[0], 'width': aug_img.shape[1],
                                              'id': new_image_index, 'file_name': json_image_file_name})
                    if after_bbs is not None:
                        for bbox in after_bbs:
                            coco_box = [bbox.x1, bbox.y1, bbox.width, bbox.height]
                            coco_box = [float(x) for x in coco_box]
                            category_id = bbox.label
                            ann = {'id': new_annotation_index, 'image_id': new_image_index, 'bbox': coco_box,
                                   'category_id': category_id, 'area': float(bbox.area), 'iscrowd': 0}
                            new_json['annotations'].append(ann)
                            new_annotation_index += 1
                    new_image_index += 1
    print(f"\ttotal new images: {new_image_index}")
    print(f"\ttotal new annotations: {new_annotation_index}")
    return new_json

def get_json_info(json_file):
    '''
    通过json文件 获得以文件名为键的字典

    输入
    json_file json文件路径
    输出
    image_info  文件名为键的字典
    origin_json_info 原始json
    '''
    with open(json_file, 'r') as f:
        origin_json_info = json.load(f)

    images = origin_json_info['images']
    annotations = origin_json_info['annotations']
    categories = origin_json_info['categories']
    image_info = {}
    for image in images:
        name = image['file_name']
        id = image['id']
        image_info[name] = {'image': image, 'annotations': []}
        for ann in annotations:
            if ann['image_id']==id:
                image_info[name]['annotations'].append(ann)
    return image_info, origin_json_info


if __name__ == "__main__":
    src_images_dir = f"/media/E_4TB/YL/mmlab/mmfewshot/data/xyb_select/images/val"      # 原图像的保存路径
    aug_img_dir = f'/media/E_4TB/YL/mmlab/mmfewshot/data/xyb_aug/images/val_aug' # 增强后图片的保存路径

    json_label_file = f"/media/E_4TB/YL/mmlab/mmfewshot/data/xyb_aug/annotations/val.json"   # 原始json
    new_json_label_file = f"/media/E_4TB/YL/mmlab/mmfewshot/data/xyb_aug/annotations/val_aug.json" # 增强后的json

    if os.path.exists(aug_img_dir):
        shutil.rmtree(aug_img_dir)
    os.mkdir(aug_img_dir)
    coco = COCO(json_label_file)
    image_info, origin_json = get_json_info(json_label_file)
    new_json = {'images': [], 'annotations': [], 'categories': origin_json['categories']}
    new_json = aug_save_img(src_images_dir, aug_img_dir, aug_times=20, image_info=image_info, new_json=new_json, new_image_pre_dir="xyb/images/val_aug/")
    with open(new_json_label_file, 'w') as f:
        json.dump(new_json, f, indent=2)

