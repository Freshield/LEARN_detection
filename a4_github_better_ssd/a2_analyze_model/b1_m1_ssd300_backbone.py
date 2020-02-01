#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b2_m1_ssd_300_backbone.py
@Time: 2020-02-05 16:19
@Last_update: 2020-02-05 16:19
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
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate
from keras.regularizers import l2
import keras.backend as K


def ssd300_backbone(image_size, l2_reg,
            subtract_mean=[123, 117, 104], divide_by_stddev=None, swap_channels=[2, 1, 0],):
    """ssd 300的主干网络"""
    ############################################################################
    # Build the network.
    ############################################################################
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################
    def identity_layer(tensor):
        # 返回tensor
        return tensor
    def input_mean_normalization(tensor):
        # 所有tensor减去各个通道的平均值，平均正则化
        return tensor - np.array(subtract_mean)
    def input_stddev_normalization(tensor):
        # 所有tensor各个通道处以标准差，标准差正则化
        return tensor / np.array(divide_by_stddev)
    def input_channel_swap(tensor):
        # 调换tensor的顺序
        if len(swap_channels) == 3:
            return K.stack(
                [tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]],
                            tensor[..., swap_channels[3]]], axis=-1)

    # 开始构建
    # 300,300,3
    x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    # 是否减去平均值
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_mean_normalization')(x1)
    # 是否除以标准差
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_stddev_normalization')(x1)
    # 是否转换通道数
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(
            x1)

    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv1_1')(x1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv1_2')(conv1_1)
    # 150,150,64
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)


    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv2_2')(conv2_1)
    # 75,75,128
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv3_3')(conv3_2)
    # 38,38,256
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv4_2')(conv4_1)
    # 38,38,512
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv4_3')(conv4_2)
    # 19,19,512
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv5_3')(conv5_2)
    # 19,19,512
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)
    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=l2(l2_reg), name='fc6')(pool5)
    # 19,19,1024
    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=l2(l2_reg), name='fc7')(fc6)

    # 19,19,256
    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv6_1')(fc7)
    # 21,21,256
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    # 10,10,512
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv6_2')(conv6_1)

    # 10,10,128
    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv7_1')(conv6_2)
    # 12,12,128
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
    # 5,5,256
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv7_2')(conv7_1)

    # 5,5,128
    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv8_1')(conv7_2)
    # 3,3,256
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv8_2')(conv8_1)

    # 3,3,128
    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv9_1')(conv8_2)
    # 1,1,256
    conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv9_2')(conv9_1)

    # conv4_3-38,38,512, fc7-19,19,1024, conv6_2-10,10,512,
    # conv7_2-5,5,256, conv8_2-3,3,256, conv9_2-1,1,256

    return x, conv4_3, fc7, conv6_2, conv7_2, conv8_2, conv9_2


if __name__ == '__main__':
    x, conv4_3, fc7, conv6_2, conv7_2, conv8_2, conv9_2 = ssd300_backbone(
        (300,300,3), 0.05)