#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b5_m1_get_predictor_size.py
@Time: 2020-01-28 18:57
@Last_update: 2020-01-28 18:57
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


def get_predictor_size(conf_tuple):
    """得到分类预测部分的shape"""
    conv4_3_norm_mbox_conf, fc7_mbox_conf, conv6_2_mbox_conf, conv7_2_mbox_conf, conv8_2_mbox_conf, conv9_2_mbox_conf = conf_tuple

    return np.array([conv4_3_norm_mbox_conf._keras_shape[1:3],
              fc7_mbox_conf._keras_shape[1:3],
              conv6_2_mbox_conf._keras_shape[1:3],
              conv7_2_mbox_conf._keras_shape[1:3],
              conv8_2_mbox_conf._keras_shape[1:3],
              conv9_2_mbox_conf._keras_shape[1:3]])