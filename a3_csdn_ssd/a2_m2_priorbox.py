#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a2_m2_priorbox.py
@Time: 2019-10-31 17:22
@Last_update: 2019-10-31 17:22
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer



class PriorBox(Layer):
    """
    用来生成先验框的类

    # 参数
    img_size: 输入图像的大小(w, h)对应着(x, y)
    min_size: 框的最小大小
    max_size: 框的最大大小
    aspect_ratio: 长宽比的大小列表
    flip: 是否要反转长宽比
    variances: x,y,w,h的差别列表
    clip: 是否让输出保持在[0,1]之间

    # 输入大小
    b,x,y,c

    # 输出大小
    b, num_boxes, 8
    """
    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, variances=[0.1], clip=True, **kwargs):
        # 指出w,h是第几维的值
        self.waxis = 1
        self.haxis = 2
        self.img_size = img_size
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]

        if aspect_ratios is not None:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    if ar == 1 or ar == 1.0:
                        self.aspect_ratios.append(ar)
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances)
        self.clip = clip
        super(PriorBox, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        # shape[-1] = self.kernel_num
        shape[1:] = [int(i) for i in self.out_shape[1:]]
        return tuple(shape)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        # shape[-1] = self.kernel_num
        shape[1:] = [int(i) for i in self.out_shape[1:]]
        return tuple(shape)

    def call(self, x, mask=None):
        # 先得到输入的尺寸
        input_shape = x._keras_shape
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        img_width = self.img_size[0]
        img_height = self.img_size[1]

        # 设定先验框到形状
        box_widths = []
        box_heights = []
        # 遍历长宽比，可能为[1, 2, 3]
        for ar in self.aspect_ratios:
            # 第一个ar为1的情况
            if ar == 1 and len(box_widths) == 0:
                # 添加长宽比最小值
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            # 代表ar为1但是不是第一个元素，对应论文中ar为1时增加到长宽比
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            # 当ar不为1时到长宽比
            else:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))
        box_widths = np.array(box_widths)
        box_heights = np.array(box_heights)

        # 定义先验框的中心点,得到x的坐标列表,y的坐标列表
        # 举例说明 img_width:300, layer_width:38
        step_x = img_width / layer_width
        step_y = img_height / layer_height
        # 生成从0.5*step_x到img_width-0.5*step_x之间的数据，一共生成layer_width个
        # 举例说明 step_x = 7.98, 也就是从3.94到296一共生成38个数据
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)
        # 生成中心点的坐标数组
        # 举例: x: [[1,2,3],[1,2,3],[1,2,3]], y: [[1,1,1],[2,2,2],[3,3,3]]
        centers_x, centers_y = np.meshgrid(linx, liny)
        # 拉平
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

        # 设置检测框到xmin,ymin,xmax,ymax
        num_priors = len(self.aspect_ratios)
        # 生成把x,y放到一起的tuple数据, (centerx, centery)得到所有中心坐标的坐标
        # 举例说明: 一共38*38个，[3.94,3.94]...[296, 296], (1444, 2)
        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        # 把每个坐标点重复,(centerx, centery, centerx, centery...)
        # 这里一共16个的原因是一共4个检验框，每个检验框需要xmin,ymin,xmax,ymax四个值
        # 举例说明: 一共38*38个, [3.94,3.94....3.94], num_priors=4, (1444, 16)
        # 把prior_boxes第一维不动,第二维堆叠2*num_priors次,达到结果为4*priors个值
        # 1444,16
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors))
        # 这里开始计算xmin,ymin,xmax,ymax
        # 每隔4个值分别减去检测框的宽度的一半得到xmin
        # 也就是说中心点的值都减去各个检验框的宽度来得到xmin,ymin,xmax,ymax
        prior_boxes[:, ::4] -= 0.5 * box_widths
        # 得到ymin
        prior_boxes[:, 1::4] -= 0.5 * box_heights
        # 得到xmax
        prior_boxes[:, 2::4] += 0.5 * box_widths
        # 得到ymax
        prior_boxes[:, 3::4] += 0.5 * box_heights
        # 进行归一化, 在这里变成了0-1之间的数值
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        # 这里后边维度的4应该代表的是xmin,ymin,xmax,ymax
        # 也就是说所有先验框的大小位置
        # 1444*4, 4
        prior_boxes = prior_boxes.reshape(-1, 4)
        # 把越界的值拉回边缘
        if self.clip:
            prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)

        # 设置variances
        num_boxes = len(prior_boxes)
        # 给每个xmin,ymin,xmax,ymax都设置好偏差, (1444*4, 4)
        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        # 把variances放到相应检验框列表中一起输出, (1444*4, 4+4)
        prior_boxes = np.concatenate((prior_boxes, variances), axis=1)

        # 转换为和输入相同大小的输出 (batch,1444*4, 8)
        prior_boxes_tensor = K.expand_dims(K.variable(prior_boxes), 0)
        pattern = [tf.shape(x)[0], 1, 1]
        prior_boxes_tensor = tf.tile(prior_boxes_tensor, pattern)

        self.out_shape = prior_boxes_tensor.shape

        return prior_boxes_tensor


