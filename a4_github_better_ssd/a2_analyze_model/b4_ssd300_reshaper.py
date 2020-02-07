#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b4_ssd300_reshaper.py
@Time: 2020-01-29 18:34
@Last_update: 2020-01-29 18:34
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


def ssd300_reshaper(conf_tuple, loc_tuple, priorbox_tuple, n_classes=21):
    """把计算完的层都进行reshape然后concat到一起"""
    ### Reshape
    # 对tuple进行解包
    conv4_3_norm_mbox_conf, fc7_mbox_conf, conv6_2_mbox_conf, conv7_2_mbox_conf, conv8_2_mbox_conf, conv9_2_mbox_conf = conf_tuple
    conv4_3_norm_mbox_loc, fc7_mbox_loc, conv6_2_mbox_loc,conv7_2_mbox_loc, conv8_2_mbox_loc, conv9_2_mbox_loc = loc_tuple
    conv4_3_norm_mbox_priorbox, fc7_mbox_priorbox, conv6_2_mbox_priorbox, conv7_2_mbox_priorbox, conv8_2_mbox_priorbox, conv9_2_mbox_priorbox = priorbox_tuple

    # 针对分类部分进行reshape
    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    # (38*38*4, 21)
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='conv4_3_norm_mbox_conf_reshape')(
        conv4_3_norm_mbox_conf)
    # (19*19*6, 21)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    # (10*10*6, 21)
    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    # (5*5*6, 21)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    # (3*3*4, 21)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    # (1*1*4, 21)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)

    # 针对loc部分进行reshape
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    # (38*38*4, 4)
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    # (19*19*6, 4)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    # (10*10*6, 4)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    # (5*5*6, 4)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    # (3*3*4, 4)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    # (1*1*4, 4)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)

    # 针对priorbox部分进行reshape
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    # (38*38*4, 8)
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(
        conv4_3_norm_mbox_priorbox)
    # (19*19*6, 8)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    # (10*10*6, 8)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    # (5*5*6, 8)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    # (3*3*4, 8)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    # (1*1*4, 8)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

    # 把各个部分分别concat到一起
    # 38*38*4+19*19*6+10*10*6+5*5*6+3*3*4+1*1*4 = 8732
    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    # (batch, 8732, 21)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    # (batch, 8732, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    # (batch, 8732, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                               fc7_mbox_priorbox_reshape,
                                                               conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape,
                                                               conv9_2_mbox_priorbox_reshape])

    return mbox_conf, mbox_loc, mbox_priorbox


if __name__ == '__main__':
    from b1_ssd300_backbone import ssd300_backbone
    from b2_ssd300_loc_conf import ssd300_loc_conf
    from b3_ssd300_priorbox import ssd300_priorbox

    x, layer_tuple = ssd300_backbone(
        (300, 300, 3))

    conf_tuple, loc_tuple = ssd300_loc_conf(layer_tuple)

    priorbox_tuple = ssd300_priorbox(loc_tuple, (300, 300, 3))

    ssd300_reshaper(conf_tuple, loc_tuple, priorbox_tuple)