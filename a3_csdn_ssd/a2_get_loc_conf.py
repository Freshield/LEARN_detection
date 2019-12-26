#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a2_priorbox.py
@Time: 2019-10-31 16:45
@Last_update: 2019-10-31 16:45
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
from a1_ssd300_model import SSD300Backbone
from normalize import Normalize
from keras.layers import Convolution2D, Flatten, Dense, Reshape, Concatenate, Activation
from keras.models import Model
from priorbox import PriorBox


def GetLocConf(net, img_size, num_classes=21):
    # 先验框的大小分别为[30, 60, 114, 168, 222, 276, 330]
    print()
    print('Begin the loc conf')
    # -------------------------------------------------------------
    # Prediction form conv4_3, (38,38,512)
    #   norm first, 让conv4_3的值在0-20之间
    #   for location
    #       通过卷积得到预测框的位置信息, 一共4个预测框, 每个预测框4个值, 所以最后是4*4=16个
    #   for confidence
    #       通过卷积得到预测框的分类信息, 一共4个预测框, 每个预测框21个类别, 所有一共是4*21=84个
    #   for prior_box
    #       这里先验框是一定的，是直接根据图像大小得到的所有设定好的先验框
    #       通过卷积得到先验框的对角坐标, 一共4个先验框, 每个先验框8个信息
    #       包括4个坐标信息和4个variance, 一共是4*8=32个, 但是其中的4放到了前边，所以后边为8维
    # -------------------------------------------------------------
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3'])
    # ----for location----
    num_priors = 4
    x = Convolution2D(num_priors * 4, 3, 3, border_mode='same',
                      name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc'] = x
    # 38,38,16
    print('conv4 mbox loc shape:', x.shape)
    net['conv4_3_norm_mbox_loc_flat'] = Flatten(name='conv4_3_norm_mbox_loc_flat')(net['conv4_3_norm_mbox_loc'])

    # ----for confidence----
    x = Convolution2D(num_priors * num_classes, 3, 3, border_mode='same',
                      name='conv4_3_norm_mbox_conf')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf'] = x
    # 38,38,84
    print('conv4 mbox conf shape:', x.shape)
    flatten = Flatten(name='conv4_3_norm_mbox_conf_flat')
    net['conv4_3_norm_mbox_conf_flat'] = flatten(net['conv4_3_norm_mbox_conf'])

    # ----for prior box----
    priorbox = PriorBox(img_size, 30., 60., aspect_ratios=[1, 2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    net['conv4_3_norm_mbox_priorbox'] = priorbox(net['conv4_3_norm'])
    # 38*38*4, 8
    # 这里前边的4代表每个像素有4个先验框，后边的8代表每个先验框的xmin,ymin,xmax,ymax以及四个variance
    # 这里的值均为0-1的小数
    print('conv4 priorbox shape:', net['conv4_3_norm_mbox_priorbox'].shape)

    # -------------------------------------------------------------
    # Prediction form fc7, (19,19,1024)
    #   for location
    #       通过卷积得到预测框的位置信息, 一共6个预测框, 每个预测框6个值, 所以最后是6*4=24个
    #   for confidence
    #       通过卷积得到预测框分类信息, 一共6个预测框, 每个预测框21个类别, 所有一共是6*21=126个
    #   for prior_box
    #       这里先验框是一定的，是直接根据图像大小得到的所有设定好的先验框
    #       通过卷积得到先验框的对角坐标, 一共6个先验框, 每个先验框8个信息
    #       包括4个坐标信息和4个variance, 一共是6*8=48个, 但是其中的6放到了前边，所以后边为8维
    # -------------------------------------------------------------
    num_priors = 6
    # ---for location---
    net['fc7_mbox_loc'] = Convolution2D(num_priors * 4, 3, 3,
                                        border_mode='same',
                                        name='fc7_mbox_loc')(net['fc7'])
    # 19,19,24
    print('fc7 mbox loc shape:', net['fc7_mbox_loc'].shape)
    flatten = Flatten(name='fc7_mbox_loc_flat')
    net['fc7_mbox_loc_flat'] = flatten(net['fc7_mbox_loc'])

    # ---for confidence---
    net['fc7_mbox_conf'] = Convolution2D(num_priors * num_classes, 3, 3,
                                         border_mode='same',
                                         name='fc7_mbox_conf')(net['fc7'])
    # 19,19,126
    print('fc7 mbox conf shape:', net['fc7_mbox_conf'].shape)
    flatten = Flatten(name='fc7_mbox_conf_flat')
    net['fc7_mbox_conf_flat'] = flatten(net['fc7_mbox_conf'])

    # ---for prior box---
    priorbox = PriorBox(img_size, 60.0, max_size=114.0, aspect_ratios=[1, 2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    net['fc7_mbox_priorbox'] = priorbox(net['fc7'])
    # 19*19*6, 8
    print('fc7 mbox priorbox shape', net['fc7_mbox_priorbox'].shape)

    # -------------------------------------------------------------
    # Prediction form conv6_2, (10,10,512)
    #   for location
    #       通过卷积得到预测框的位置信息, 一共6个预测框, 每个预测框6个值, 所以最后是6*4=24个
    #   for confidence
    #       通过卷积得到预测框的分类信息, 一共6个预测框, 每个预测框21个类别, 所有一共是6*21=126个
    #   for prior_box
    #       这里先验框是一定的，是直接根据图像大小得到的所有设定好的先验框
    #       通过卷积得到先验框的对角坐标, 一共6个先验框, 每个先验框8个信息
    #       包括4个坐标信息和4个variance, 一共是6*8=48个, 但是其中的6放到了前边，所以后边为8维
    # -------------------------------------------------------------
    num_priors = 6
    # ---for location---
    x = Convolution2D(num_priors * 4, 3, 3, border_mode='same',
                      name='conv6_2_mbox_loc')(net['conv6_2'])
    # 10,10,24
    print('conv6_2 mbox loc shape:', x.shape)
    net['conv6_2_mbox_loc'] = x
    flatten = Flatten(name='conv6_2_mbox_loc_flat')
    net['conv6_2_mbox_loc_flat'] = flatten(net['conv6_2_mbox_loc'])

    # ---for confidence---
    x = Convolution2D(num_priors * num_classes, 3, 3, border_mode='same',
                      name='conv6_2_mbox_conf')(net['conv6_2'])
    # 10,10,126
    print('conv6_2 mbox conf shape:', x.shape)
    net['conv6_2_mbox_conf'] = x
    flatten = Flatten(name='conv6_2_mbox_conf_flat')
    net['conv6_2_mbox_conf_flat'] = flatten(net['conv6_2_mbox_conf'])

    # ---for prior box---
    priorbox = PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[1, 2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_2_mbox_priorbox')
    net['conv6_2_mbox_priorbox'] = priorbox(net['conv6_2'])
    # 10*10*6, 8
    print('conv6_2 mbox priorbox shape:', net['conv6_2_mbox_priorbox'].shape)

    # -------------------------------------------------------------
    # Prediction form conv7_2, (5,5,256)
    #   for location
    #       通过卷积得到预测框的位置信息, 一共6个预测框, 每个预测框6个值, 所以最后是6*4=24个
    #   for confidence
    #       通过卷积得到预测框的分类信息, 一共6个预测框, 每个预测框21个类别, 所有一共是6*21=126个
    #   for prior_box
    #       这里先验框是一定的，是直接根据图像大小得到的所有设定好的先验框
    #       通过卷积得到先验框的对角坐标, 一共6个先验框, 每个先验框8个信息
    #       包括4个坐标信息和4个variance, 一共是6*8=48个, 但是其中的6放到了前边，所以后边为8维
    # -------------------------------------------------------------
    num_priors = 6
    # ---for location---
    x = Convolution2D(num_priors * 4, 3, 3, border_mode='same',
                      name='conv7_2_mbox_loc')(net['conv7_2'])
    # 5,5,24
    print('conv7_2 mbox loc shape:', x.shape)
    net['conv7_2_mbox_loc'] = x
    flatten = Flatten(name='conv7_2_mbox_loc_flat')
    net['conv7_2_mbox_loc_flat'] = flatten(net['conv7_2_mbox_loc'])

    # ---for confidence---
    x = Convolution2D(num_priors * num_classes, 3, 3, border_mode='same',
                      name='conv7_2_mbox_conf')(net['conv7_2'])
    # 5,5,126
    print('conv7_2 mbox loc shape:', x.shape)
    net['conv7_2_mbox_conf'] = x
    flatten = Flatten(name='conv7_2_mbox_conf_flat')
    net['conv7_2_mbox_conf_flat'] = flatten(net['conv7_2_mbox_conf'])

    # ---for prior box---
    priorbox = PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[1, 2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_2_mbox_priorbox')
    net['conv7_2_mbox_priorbox'] = priorbox(net['conv7_2'])
    # 5*5*6, 8
    print('conv7_2 mbox priorbox shape:', net['conv7_2_mbox_priorbox'].shape)

    # -------------------------------------------------------------
    # Prediction form conv8_2, (3,3,256)
    #   for location
    #       通过卷积得到预测框的位置信息, 一共4个预测框, 每个预测框4个值, 所以最后是4*4=16个
    #   for confidence
    #       通过卷积得到预测框的分类信息, 一共4个预测框, 每个预测框21个类别, 所有一共是4*21=84个
    #   for prior_box
    #       这里先验框是一定的，是直接根据图像大小得到的所有设定好的先验框
    #       通过卷积得到先验框的对角坐标, 一共4个先验框, 每个先验框8个信息
    #       包括4个坐标信息和4个variance, 一共是4*8=32个, 但是其中的4放到了前边，所以后边为8维
    # -------------------------------------------------------------
    num_priors = 4
    # ---for location---
    x = Convolution2D(num_priors * 4, 3, 3, border_mode='same',
                      name='conv8_2_mbox_loc')(net['conv8_2'])
    # 3,3,16
    print('conv8_2 mbox loc shape:', x.shape)
    net['conv8_2_mbox_loc'] = x
    flatten = Flatten(name='conv8_2_mbox_loc_flat')
    print(x.shape)
    net['conv8_2_mbox_loc_flat'] = flatten(net['conv8_2_mbox_loc'])
    print(net['conv8_2_mbox_loc_flat'].shape)

    # ---for confidence---
    x = Convolution2D(num_priors * num_classes, 3, 3, border_mode='same',
                      name='conv8_2_mbox_conf')(net['conv8_2'])
    # 3,3,84
    print('conv8_2 mbox conf shape:', x.shape)
    net['conv8_2_mbox_conf'] = x
    flatten = Flatten(name='conv8_2_mbox_conf_flat')
    net['conv8_2_mbox_conf_flat'] = flatten(net['conv8_2_mbox_conf'])

    # ---for prior box---
    priorbox = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[1, 2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')
    net['conv8_2_mbox_priorbox'] = priorbox(net['conv8_2'])
    # 3*3*4, 8
    print('conv8_2 mbox priorbox shape:', net['conv8_2_mbox_priorbox'].shape)

    # -------------------------------------------------------------
    # Prediction form pool6, (256)
    #   for location
    #       通过卷积得到预测框的位置信息, 一共4个预测框, 每个预测框4个值, 所以最后是4*4=16个
    #   for confidence
    #       通过卷积得到预测框的分类信息, 一共4个预测框, 每个预测框21个类别, 所有一共是4*21=84个
    #   for prior_box
    #       这里先验框是一定的，是直接根据图像大小得到的所有设定好的先验框
    #       通过卷积得到先验框的对角坐标, 一共4个先验框, 每个先验框8个信息
    #       包括4个坐标信息和4个variance, 一共是4*8=32个, 但是其中的4放到了前边，所以后边为8维
    # -------------------------------------------------------------
    num_priors = 4
    # ---for location---
    x = Dense(num_priors * 4, name='pool6_mbox_loc_flat')(net['pool6'])
    # 16
    print('pool6 mbox loc flat shape:', x.shape)
    net['pool6_mbox_loc_flat'] = x

    # ---for confidence---
    x = Dense(num_priors * num_classes, name='pool6_mbox_conf_flat')(net['pool6'])
    # 84
    print('pool6 mbox conf flat shape:', x.shape)
    net['pool6_mbox_conf_flat'] = x

    # ---for prior box---
    # 因为PriorBox需要三维的数据,所以这里需要先reshape下
    net['pool6_reshaped'] = Reshape((1, 1, 256),
                                    name='pool6_reshaped')(net['pool6'])
    priorbox = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[1, 2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='pool6_mbox_priorbox')
    net['pool6_mbox_priorbox'] = priorbox(net['pool6_reshaped'])
    # 1*1*4, 8
    print('pool6 mbox priorbox shape:', net['pool6_mbox_priorbox'].shape)

    # 把所有结果放到一起
    # ---for location---
    # 38*38*4*4 + 19*19*6*4 + 10*10*6*4 + 5*5*6*4 + 3*3*4*4 + 1*1*4*4 = 34928
    net['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')(
        [net['conv4_3_norm_mbox_loc_flat'], net['fc7_mbox_loc_flat'],
         net['conv6_2_mbox_loc_flat'], net['conv7_2_mbox_loc_flat'],
         net['conv8_2_mbox_loc_flat'], net['pool6_mbox_loc_flat']])

    # ---for confidence---
    # 38*38*4*21 + 19*19*6*21 + 10*10*6*21 + 5*5*6*21 + 3*3*4*21 + 1*1*4*21 = 183372
    net['mbox_conf'] = Concatenate(axis=1, name='mbox_conf')(
        [net['conv4_3_norm_mbox_conf_flat'], net['fc7_mbox_conf_flat'],
         net['conv6_2_mbox_conf_flat'], net['conv7_2_mbox_conf_flat'],
         net['conv8_2_mbox_conf_flat'], net['pool6_mbox_conf_flat']])

    # ---for prior box---
    # 38*38*4 + 19*19*6 + 10*10*6 + 5*5*6 + 3*3*4 + 1*1*4 = 8732, (8732, 8)
    net['mbox_priorbox'] = Concatenate(axis=1, name='mbox_priorbox')(
        [net['conv4_3_norm_mbox_priorbox'], net['fc7_mbox_priorbox'],
         net['conv6_2_mbox_priorbox'], net['conv7_2_mbox_priorbox'],
         net['conv8_2_mbox_priorbox'], net['pool6_mbox_priorbox']])

    # 计算出一共多少个检验框
    # num_boxes = 38*38*4 + 19*19*6 + 10*10*6 + 5*5*6 + 3*3*4 + 1*1*4 = 8732
    num_boxes = net['mbox_loc']._keras_shape[-1] // 4
    # ---for location---
    # 8732, 4
    net['mbox_loc'] = Reshape((num_boxes, 4),
                              name='mbox_loc_final')(net['mbox_loc'])

    # ---for confidence---
    # 8732, 21
    net['mbox_conf'] = Reshape((num_boxes, num_classes),
                               name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = Activation('softmax',
                                  name='mbox_conf_final')(net['mbox_conf'])

    # 融合到一起
    net['predictions'] = Concatenate(axis=2, name='predictions')(
        [net['mbox_loc'], net['mbox_conf'], net['mbox_priorbox']])

    # 8732, 4+21+8
    print('Final gather shape', net['predictions'].shape)

    model = Model(inputs=net['input'], outputs=net['predictions'])

    return model, net


if __name__ == '__main__':
    from keras.utils import plot_model
    net = SSD300Backbone((300,300,3))
    model, net = GetLocConf(net, (300,300))
    # model.summary()
    # plot_model(model, 'data/model.png')

    # import pickle
    # config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # sess.run(tf.global_variables_initializer())
    # mbox_priorbox = sess.run(net['mbox_priorbox'], feed_dict={net['input']: np.zeros((1,300,300,3))})[0]
    # print(mbox_priorbox.shape)
    # print(mbox_priorbox[0])
    # pickle.dump(mbox_priorbox, open('data/prior_boxes_ssd300.pkl', 'wb'))