#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a3_ssd_layer.py
@Time: 2019-10-31 16:50
@Last_update: 2019-10-31 16:50
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import keras
import numpy as np
import keras.backend as K
from keras.layers import Layer, InputSpec


class Normalize(Layer):
    def __init__(self, scale, **kwargs):
        self.axis = 3
        self.scale = scale
        super(Normalize, self). __init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = K.variable(init_gamma, name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]

    def call(self, x, **kwargs):
        output = K.l2_normalize(x, self.axis)
        output *= self.gamma
        return output