#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :heatmap_visualization.py
# @Time      :2022/3/11 ??9:46
# @Author    :Yangliang


import cv2
import numpy as np
import os

def draw_heatmap(features, img_metas, save_dir):
    assert isinstance(features, tuple) and len(features) == 1
    features = features[0]
    assert len(img_metas) == features.shape[0]
    features = features.detach().cpu().numpy()
    heatmaps = np.sum(features, axis=1)
    heatmaps = np.maximum(heatmaps, 0)

    for i in range(len(img_metas)):
        heatmap = heatmaps[i, :, :]
        heatmap /= np.max(heatmap)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img = cv2.imread(img_metas[i]['filename'])
        h_size = img_metas[i]['ori_shape'][1]
        w_size = img_metas[i]['ori_shape'][0]
        heatmap = cv2.resize(heatmap, (h_size, w_size))
        superimposed_img = cv2.addWeighted(img, 0.4, heatmap, 0.6, 0)
        img_cat = np.concatenate((superimposed_img, heatmap, img), axis=1)
        cv2.imwrite(os.path.join(save_dir, os.path.basename(img_metas[i]['ori_filename'])), img_cat)