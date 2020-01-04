#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a5_generator.py
@Time: 2019-11-01 16:49
@Last_update: 2019-11-01 16:49
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import os
import numpy as np
from random import shuffle
# from scipy.misc import imread, imresize
from imageio import imread
from skimage.transform import resize as imresize
from keras.applications.imagenet_utils import preprocess_input


class Generator(object):
    """
    生成器类

    """
    def __init__(self, gt, bbox_util,
                 batch_size, path_prefix,
                 train_keys, val_keys, image_size,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3. / 4., 4. / 3.]):
        # 为VOC数据记得label信息
        self.gt = gt
        # 为bbox的类别
        self.bbox_util = bbox_util
        # batch size
        self.batch_size = batch_size
        # 路径前缀，文件夹位置
        self.path_prefix = path_prefix
        # 训练的图像名称列表，为gt中的key值，也就是图像名称
        self.train_keys = train_keys
        # 验证的图像名称列表，为gt中的key值，也就是图像名称
        self.val_keys = val_keys
        # 训练的总数量
        self.train_batches = len(train_keys)
        # 验证的总数量
        self.val_batches = len(val_keys)
        # 图片大小
        self.image_size = image_size
        # 颜色抖动方法列表
        self.color_jitter = []
        # 饱和度抖动
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        # 亮度抖动
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        # 对比度抖动
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        # 光度标准化
        self.lighting_std = lighting_std
        # 横向反转
        self.hflip_prob = hflip_prob
        # 纵向反转
        self.vflip_prob = vflip_prob
        # 是否进行裁剪
        self.do_crop = do_crop
        # 裁剪区域列表
        self.crop_area_range = crop_area_range
        # 缩放比例列表
        self.aspect_ratio_range = aspect_ratio_range

    # 彩图变灰度图
    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    # 饱和度变化
    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    # 亮度变化
    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    # 对比度变化
    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    # 光度变化
    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)

    # 水平反转
    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y

    # 垂直反转
    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y

    # 随机大小裁剪
    def random_sized_crop(self, img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w_rel = w / img_w
        w = int(w)
        h = min(h, img_h)
        h_rel = h / img_h
        h = int(h)
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y + h, x:x + w]
        new_targets = []
        for box in targets:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                    y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
        return img, new_targets

    # 生成器生成部分
    def generate(self, train=True):
        while True:
            # 首先随机打乱顺序
            if train:
                shuffle(self.train_keys)
                keys = self.train_keys
            else:
                shuffle(self.val_keys)
                keys = self.val_keys

            # 输入的图像列表
            inputs = []
            # 输入的label列表
            targets = []
            # 遍历所有的图像键
            for key in keys:
                # 读取图和label
                img_path = os.path.join(self.path_prefix, key)
                img = imread(img_path).astype('float32')
                y = self.gt[key].copy()
                # 如果是训练且裁剪，则岁键裁剪
                if train and self.do_crop:
                    img, y = self.random_sized_crop(img, y)
                # 转换为默认图像大小
                img = imresize(img, self.image_size).astype('float32')
                # 进行数据增强
                if train:
                    # 重新排序颜色抖动方法列表
                    shuffle(self.color_jitter)
                    # 进行数据抖动
                    for jitter in self.color_jitter:
                        img = jitter(img)
                    # 进行光度归一
                    if self.lighting_std:
                        img = self.lighting(img)
                    # 进行水平反转
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    # 进行垂直反转
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)
                # 使用bbox来把原始的label转换为训练使用的label
                y = self.bbox_util.assign_boxes(y)
                # 数据放入相应的列表
                inputs.append(img)
                targets.append(y)
                # 如果到达一个batch的大小，则转换为矩阵，yield出去
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    # 这里的preprocess主要是进行图像数据的归一化操作
                    yield preprocess_input(tmp_inp), tmp_targets


if __name__ == '__main__':
    import pickle
    gt = pickle.load(open('data/VOC2007.p', 'rb'))
    keys = sorted(gt.keys())
    num_train = int(round(0.8 * len(keys)))
    train_keys = keys[:num_train]
    val_keys = keys[num_train:]
    num_val = len(val_keys)

    from a4_bbox import BBoxUtility
    priors = pickle.load(open('data/prior_boxes_ssd300.pkl', 'rb'))
    bbox_util = BBoxUtility(21, priors)

    path_prefix = '/media/freshield/SSD_1T/Data/a13_detection/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'
    gen = Generator(gt, bbox_util, 16, path_prefix,
                    train_keys, val_keys,
                    (300,300), do_crop=False)

    img, y = gen.generate().__next__()
    print(img.shape)
    print(y.shape)
