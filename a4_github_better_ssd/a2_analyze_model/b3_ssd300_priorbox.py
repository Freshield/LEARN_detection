#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b3_ssd300_priorbox.py
@Time: 2020-02-06 13:39
@Last_update: 2020-02-06 13:39
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
from b3_m1_anchorboxes import AnchorBoxes


def ssd300_priorbox(loc_tuple, image_size,
                    scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                    aspect_ratios=[[1.0, 2.0, 0.5],
                                   [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                   [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                   [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                   [1.0, 2.0, 0.5],
                                   [1.0, 2.0, 0.5]],
                    steps=[8, 16, 32, 64, 100, 300],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    variances=np.array([0.1, 0.1, 0.2, 0.2]),
                    two_boxes_for_ar1=True,
                    clip_boxes=False, coords='centroids', normalize_coords=True):
    """得到相应层的先验框"""
    # 长宽通道
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]
    # 得到相应的层
    conv4_3_norm_mbox_loc, fc7_mbox_loc, conv6_2_mbox_loc,conv7_2_mbox_loc, conv8_2_mbox_loc, conv9_2_mbox_loc = loc_tuple

    # 生成先验框部分
    # 这里只用到了loc层的shape信息
    ### Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)
    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    # (batch, 38, 38, 4, 8)
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
    # (batch, 19, 19, 6, 8)
    fc7_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1], clip_boxes=clip_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords, name='fc7_mbox_priorbox')(fc7_mbox_loc)
    # (batch, 10, 10, 6, 8)
    conv6_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
    # (batch, 5, 5, 6, 8)
    conv7_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
    # (batch, 3, 3, 4, 8)
    conv8_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4], this_offsets=offsets[4], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
    # (batch, 1, 1, 4, 8)
    conv9_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5], this_offsets=offsets[5], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)

    return conv4_3_norm_mbox_priorbox, fc7_mbox_priorbox, conv6_2_mbox_priorbox, \
           conv7_2_mbox_priorbox, conv8_2_mbox_priorbox, conv9_2_mbox_priorbox


if __name__ == '__main__':
    from b1_ssd300_backbone import ssd300_backbone
    from b2_ssd300_loc_conf import ssd300_loc_conf

    x, layer_tuple = ssd300_backbone(
        (300, 300, 3))

    conf_tuple, loc_tuple = ssd300_loc_conf(layer_tuple)

    rst = ssd300_priorbox(loc_tuple, (300, 300, 3), normalize_coords=False)

    print(rst)