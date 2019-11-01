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
    print('Begin the loc conf')
    # Prediction form conv4_3
    #     for location
    #     norm first, 让conv4_3的值在0-20之间
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3'])
    num_priors = 4
    x = Convolution2D(num_priors * 4, 3, 3, border_mode='same',
                      name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc'] = x
    print('conv4 mbox loc shape:', x.shape)
    net['conv4_3_norm_mbox_loc_flat'] = Flatten(name='conv4_3_norm_mbox_loc_flat')(net['conv4_3_norm_mbox_loc'])
    #    for conf
    x = Convolution2D(num_priors * num_classes, 3, 3, border_mode='same',
                      name='conv4_3_norm_mbox_conf')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf'] = x
    print('conv4 mbox conf shape:', x.shape)
    flatten = Flatten(name='conv4_3_norm_mbox_conf_flat')
    net['conv4_3_norm_mbox_conf_flat'] = flatten(net['conv4_3_norm_mbox_conf'])
    #    for prior box
    priorbox = PriorBox(img_size, 30., 60., aspect_ratios=[1, 2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    net['conv4_3_norm_mbox_priorbox'] = priorbox(net['conv4_3_norm'])
    print('conv4 priorbox shape:', net['conv4_3_norm_mbox_priorbox'].shape)

    # Prediction from fc7
    #   for location
    num_priors = 6
    net['fc7_mbox_loc'] = Convolution2D(num_priors * 4, 3, 3,
                                        border_mode='same',
                                        name='fc7_mbox_loc')(net['fc7'])
    print('fc7 mbox loc shape:', net['fc7_mbox_loc'].shape)
    flatten = Flatten(name='fc7_mbox_loc_flat')
    net['fc7_mbox_loc_flat'] = flatten(net['fc7_mbox_loc'])
    #   for conf
    net['fc7_mbox_conf'] = Convolution2D(num_priors * num_classes, 3, 3,
                                         border_mode='same',
                                         name='fc7_mbox_conf')(net['fc7'])
    print('fc7 mbox conf shape:', net['fc7_mbox_conf'].shape)
    flatten = Flatten(name='fc7_mbox_conf_flat')
    net['fc7_mbox_conf_flat'] = flatten(net['fc7_mbox_conf'])
    #   for priorbox
    priorbox = PriorBox(img_size, 60.0, max_size=114.0, aspect_ratios=[1, 2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    net['fc7_mbox_priorbox'] = priorbox(net['fc7'])
    print('fc7 mbox priorbox shape', net['fc7_mbox_priorbox'].shape)

    # Prediction from conv6_2
    num_priors = 6
    x = Convolution2D(num_priors * 4, 3, 3, border_mode='same',
                      name='conv6_2_mbox_loc')(net['conv6_2'])
    print('conv6_2 mbox loc shape:', x.shape)
    net['conv6_2_mbox_loc'] = x
    flatten = Flatten(name='conv6_2_mbox_loc_flat')
    net['conv6_2_mbox_loc_flat'] = flatten(net['conv6_2_mbox_loc'])
    x = Convolution2D(num_priors * num_classes, 3, 3, border_mode='same',
                      name='conv6_2_mbox_conf')(net['conv6_2'])
    print('conv6_2 mbox conf shape:', x.shape)
    net['conv6_2_mbox_conf'] = x
    flatten = Flatten(name='conv6_2_mbox_conf_flat')
    net['conv6_2_mbox_conf_flat'] = flatten(net['conv6_2_mbox_conf'])
    priorbox = PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[1, 2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_2_mbox_priorbox')
    net['conv6_2_mbox_priorbox'] = priorbox(net['conv6_2'])
    print('conv6_2 mbox priorbox shape:', net['conv6_2_mbox_priorbox'].shape)

    # Prediction from conv7_2
    num_priors = 6
    x = Convolution2D(num_priors * 4, 3, 3, border_mode='same',
                      name='conv7_2_mbox_loc')(net['conv7_2'])
    print('conv7_2 mbox loc shape:', x.shape)
    net['conv7_2_mbox_loc'] = x
    flatten = Flatten(name='conv7_2_mbox_loc_flat')
    net['conv7_2_mbox_loc_flat'] = flatten(net['conv7_2_mbox_loc'])
    x = Convolution2D(num_priors * num_classes, 3, 3, border_mode='same',
                      name='conv7_2_mbox_conf')(net['conv7_2'])
    print('conv7_2 mbox loc shape:', x.shape)
    net['conv7_2_mbox_conf'] = x
    flatten = Flatten(name='conv7_2_mbox_conf_flat')
    net['conv7_2_mbox_conf_flat'] = flatten(net['conv7_2_mbox_conf'])
    priorbox = PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[1, 2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_2_mbox_priorbox')
    net['conv7_2_mbox_priorbox'] = priorbox(net['conv7_2'])
    print('conv7_2 mbox priorbox shape:', net['conv7_2_mbox_priorbox'].shape)

    # Prediction from conv8_2
    num_priors = 4
    x = Convolution2D(num_priors * 4, 3, 3, border_mode='same',
                      name='conv8_2_mbox_loc')(net['conv8_2'])
    print('conv8_2 mbox loc shape:', x.shape)
    net['conv8_2_mbox_loc'] = x
    flatten = Flatten(name='conv8_2_mbox_loc_flat')
    print(x.shape)
    net['conv8_2_mbox_loc_flat'] = flatten(net['conv8_2_mbox_loc'])
    print(net['conv8_2_mbox_loc_flat'].shape)
    x = Convolution2D(num_priors * num_classes, 3, 3, border_mode='same',
                      name='conv8_2_mbox_conf')(net['conv8_2'])
    print('conv8_2 mbox conf shape:', x.shape)
    net['conv8_2_mbox_conf'] = x
    flatten = Flatten(name='conv8_2_mbox_conf_flat')
    net['conv8_2_mbox_conf_flat'] = flatten(net['conv8_2_mbox_conf'])
    priorbox = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[1, 2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')
    net['conv8_2_mbox_priorbox'] = priorbox(net['conv8_2'])
    print('conv8_2 mbox priorbox shape:', net['conv8_2_mbox_priorbox'].shape)

    # Prediction from pool6
    num_priors = 4
    x = Dense(num_priors * 4, name='pool6_mbox_loc_flat')(net['pool6'])
    print('pool6 mbox loc flat shape:', x.shape)
    net['pool6_mbox_loc_flat'] = x
    x = Dense(num_priors * num_classes, name='pool6_mbox_conf_flat')(net['pool6'])
    print('pool6 mbox conf flat shape:', x.shape)
    net['pool6_mbox_conf_flat'] = x
    net['pool6_reshaped'] = Reshape((1, 1, 256),
                                    name='pool6_reshaped')(net['pool6'])
    priorbox = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[1, 2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='pool6_mbox_priorbox')
    net['pool6_mbox_priorbox'] = priorbox(net['pool6_reshaped'])
    print('pool6 mbox priorbox shape:', net['pool6_mbox_priorbox'].shape)

    # 把所有结果放到一起
    net['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')(
        [net['conv4_3_norm_mbox_loc_flat'], net['fc7_mbox_loc_flat'],
         net['conv6_2_mbox_loc_flat'], net['conv7_2_mbox_loc_flat'],
         net['conv8_2_mbox_loc_flat'], net['pool6_mbox_loc_flat']])
    net['mbox_conf'] = Concatenate(axis=1, name='mbox_conf')(
        [net['conv4_3_norm_mbox_conf_flat'], net['fc7_mbox_conf_flat'],
         net['conv6_2_mbox_conf_flat'], net['conv7_2_mbox_conf_flat'],
         net['conv8_2_mbox_conf_flat'], net['pool6_mbox_conf_flat']])
    net['mbox_priorbox'] = Concatenate(axis=1, name='mbox_priorbox')(
        [net['conv4_3_norm_mbox_priorbox'], net['fc7_mbox_priorbox'],
         net['conv6_2_mbox_priorbox'], net['conv7_2_mbox_priorbox'],
         net['conv8_2_mbox_priorbox'], net['pool6_mbox_priorbox']])

    # 计算出一共多少个检验框
    num_boxes = net['mbox_loc']._keras_shape[-1] // 4
    net['mbox_loc'] = Reshape((num_boxes, 4),
                              name='mbox_loc_final')(net['mbox_loc'])
    net['mbox_conf'] = Reshape((num_boxes, num_classes),
                               name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = Activation('softmax',
                                  name='mbox_conf_final')(net['mbox_conf'])

    net['predictions'] = Concatenate(axis=2, name='predictions')(
        [net['mbox_loc'], net['mbox_conf'], net['mbox_priorbox']])

    print('Final gather shape', net['predictions'].shape)

    model = Model(inputs=net['input'], outputs=net['predictions'])

    return model







if __name__ == '__main__':
    from keras.utils import plot_model
    net = SSD300Backbone((300,300,3))
    model = GetLocConf(net, (300,300))
    model.summary()
    plot_model(model, 'data/model.png')