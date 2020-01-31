#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b1_m2_ssd300_loc_conf.py
@Time: 2020-02-05 17:06
@Last_update: 2020-02-05 17:06
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate
from b1_m3_l2norm_layer import L2Normalization
from keras.regularizers import l2


def ssd300_loc_conf(layer_tuple, n_boxes=[4, 6, 6, 6, 4, 4], n_classes=21, l2_reg=0.0005):
    """得到ssd300的loc和conf层"""

    conv4_3, fc7, conv6_2, conv7_2, conv8_2, conv9_2 = layer_tuple
    # Feed conv4_3 into the L2 normalization layer
    # 首先对conv4_3的值进行l2正则
    conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)

    ### Build the convolutional predictor layers on top of the base network

    # We precidt `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
    # 38,38,4*21
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    # 19,19,6*21
    fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=l2(l2_reg), name='fc7_mbox_conf')(fc7)
    # 10,10,6*21
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_conf')(conv6_2)
    # 5,5,6*21
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_conf')(conv7_2)
    # 3,3,4*21
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf')(conv8_2)
    # 1,1,4*21
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf')(conv9_2)

    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    # 38,38,4*4
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    # 19,19,6*4
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(l2_reg), name='fc7_mbox_loc')(fc7)
    # 10,10,6*4
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_loc')(conv6_2)
    # 5,5,6*4
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_loc')(conv7_2)
    # 3,3,4*4
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc')(conv8_2)
    # 1,1,4*4
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc')(conv9_2)

    return (conv4_3_norm_mbox_conf, conv6_2_mbox_conf, conv7_2_mbox_conf,
            conv8_2_mbox_conf, conv9_2_mbox_conf), \
           (conv4_3_norm_mbox_loc, conv6_2_mbox_loc, conv7_2_mbox_loc,
            conv8_2_mbox_loc, conv9_2_mbox_loc)




if __name__ == '__main__':
    from b1_m1_ssd300_backbone import ssd300_backbone

    x, layer_tuple = ssd300_backbone(
        (300, 300, 3))

    ssd300_loc_conf(layer_tuple)

