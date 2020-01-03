#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a6_losses.py
@Time: 2019-12-26 11:39
@Last_update: 2019-12-26 11:39
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import tensorflow as tf
# tf.enable_eager_execution()


class MultiboxLoss(object):
    """
    统一计算loss的类
    # 参数
        num_classes: 包括背景在内的类别数
        alpha: L1-smooth损失的权重
        neg_pos_ratio: loss中负样本和正样本的最大比率
        background_label_id: 背景类别的id
        negatives_for_hard: 当一个batch没有正样本时，多少负样本数量会来使用

    # References
        https://arxiv.org/abs/1512.02325

    # TODO
        Add possibility for background label id be not zero
    """
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=100.0):
        # 类别数量
        self.num_classes = num_classes
        # L1-smooth权重
        self.alpha = alpha
        # 正负样本比例
        self.neg_pos_ratio = neg_pos_ratio
        # 背景类别位置
        if background_label_id != 0:
            raise Exception('Only 0 as background label id is supported')
        self.background_label_id = background_label_id
        # 难例负样本数量
        self.negatives_for_hard = negatives_for_hard

    def _l1_smooth_loss(self, y_true, y_pred):
        """
        计算位置信息的L1-smooth损失

        # Arguments
            y_true: label的结果,
                tensor of shape (?, 7832, 4).
            y_pred: 预测的结果,
                tensor of shape (?, 7832, 4).

        # Returns
            l1_loss: L1-smooth loss, tensor of shape (?, 7832).

        # References
            https://arxiv.org/abs/1504.08083
        """
        # |x| < 1, 0.5*(x^2)
        # otherwise, |x| - 0.5
        # 位置的差的绝对值
        abs_loss = tf.abs(y_true - y_pred)
        # 平方的值0.5*(diff^2)
        sq_loss = 0.5 * (y_true - y_pred)**2
        # 通过where来进行选择使用那个值
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        # 把4个坐标的损失值相加返回
        return tf.reduce_sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        """计算softmax损失

        # Arguments
            y_true: label结果,
                tensor of shape (?, 7832, 21).
            y_pred: 预测结果,
                tensor of shape (?, 7832, 21).

        # Returns
            softmax_loss: Softmax loss, tensor of shape (?, 7832).
        """
        # 计算交叉熵损失
        y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)
        softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred),
                                      axis=-1)
        return softmax_loss

    def compute_loss(self, y_true, y_pred):
        # 在keras中自定义loss函数，它的两个输入必须为预测的输出和标准的输出
        # 变量：
        # y_pred: 它的shape为： (?, 8732, 4 + 21 + 8). 就是在model框架部分介绍的输出。
        # y_truth：它的shape和y_pred的shape是一样的，就是assignment那一块的输出
        # 返回最终的所有loss总和
        # 得到batch的大小
        batch_size = tf.shape(y_true)[0]
        # 得到先验框的数量, 8732
        num_boxes = tf.to_double(tf.shape(y_true)[1])
        # 计算出所有default box的loss
        # 这里是类别的损失
        conf_loss = self._softmax_loss(y_true[:, :, 4:-8],y_pred[:, :, 4:-8])
        # 这里是位置的l1损失
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4],y_pred[:, :, :4])

        # 计算正样本的loss
        # num_pos 为一个一维的array：len(num_pos)=batch
        # 这个表示哪些位置是和先验框匹配上的，也就是IOU大于阈值的先验框
        # 而这里把所有的都加起来了表示一共有多少个正例
        # (batch,)
        num_pos = tf.reduce_sum(y_true[:, :, -8], axis=-1)
        ##只需计算存在gt_box与其对应的loss
        # 通过相乘，得到所有大于阈值的先验框，也就是正例
        pos_loc_loss = tf.reduce_sum(loc_loss * y_true[:, :, -8],axis = 1)
        pos_conf_loss = tf.reduce_sum(conf_loss * y_true[:, :, -8],axis = 1)

        # 计算负样本的loss，只计算了confidence loss
        # 这里看下负例按照比率乘以正例的数据后的值和总提负例比起来哪个小，哪个小就使用哪个值，主要是防止溢出
        # (batch,)
        num_pos = tf.cast(num_pos, tf.double)
        num_neg = tf.minimum(self.neg_pos_ratio * num_pos, num_boxes - num_pos)
        # 找到所有负例数量大于0的位置，(batch,), bool
        pos_num_neg_mask = tf.greater(num_neg, 0)
        # 看刚才的结果是否有大于0的位置，如果有则为1否则则为0
        has_min = tf.to_float(tf.reduce_any(pos_num_neg_mask))
        # 这里是重新生成负例的数量，如果刚才看负例结果都没有的话
        # 这一步自动给每个batch分配100个负例的数量, (batch,)
        num_neg = tf.concat(axis=0, values=[num_neg,[(1 - has_min) * self.negatives_for_hard]])
        # tf.boolen_mask(a,b)，例如b=[true, false],a=[[[2,2],[2,3]]],则输出为[2,2]。
        # 实际上就是取num_neg为正数的那些元素，然后再在其中取num_neg中的最小的元素作为num_neg_batch。
        # 这里是找到所有负例数量中负例数量最小的值作为num_neg_batch的值
        num_neg_batch = tf.reduce_min(tf.boolean_mask(num_neg,tf.greater(num_neg, 0)))
        num_neg_batch = tf.to_int32(num_neg_batch)
        # 这里是分类类别开始的位置和结束的位置
        confs_start = 4 + self.background_label_id + 1
        confs_end = confs_start + self.num_classes - 1
        # 找到所有预测的分类结果中最大的值,max_confs的shape为：(batch, 8732)
        max_confs = tf.reduce_max(y_pred[:, :, confs_start:confs_end],axis = 2)

        # 这里主要做的是得到所有数据都铺平后，要计算的负例的索引值
        # 返回负样本的top-K个元素,最终返回的indices的shape为(batch, K=num_neg_batch)
        # 1 - y_true[:, :, -8]表示我只找所有的负样本，(batch, 8732)
        # 再乘以max_conf表示找到这些负样本的类别预测的值，(batch, 8732)
        # 最后再从中挑选负样本中前K个负样本，(batch, K)
        _, indices = tf.nn.top_k(max_confs * (1 - y_true[:, :, -8]), k=num_neg_batch)
        # 创建一个shape也为(batch,num_neg_batch)的indices
        # 这里生成的batch_idx把每层的序号生成了一遍，比如batch_idx[0]中的值都为0
        # 得到的是每个batch的起始值序号
        batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
        batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
        # tf.reshape(batch_idx, [-1])为batch*num_neg_batch
        # 再加上indices，就是当把所有损失值铺平时，要使用的负例的序号
        full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(num_boxes) +tf.reshape(indices, [-1]))
        # 把得到的conf_loss也reshape成一维，然后用full_indices对其进行取值
        neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]),full_indices)
        # 最终把负样本的confidence loss reshape 成(batch, num_neg_batch),再对每个sample上的loss求和。
        neg_conf_loss = tf.reshape(neg_conf_loss,[batch_size, num_neg_batch])
        neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=1)

        # 整合所有的loss：positive loss 和 negative loss
        # 这里是分类损失值
        total_loss = pos_conf_loss + neg_conf_loss
        num_pos = tf.to_float(num_pos)
        num_neg_batch = tf.to_float(num_neg_batch)
        total_loss = tf.to_float(total_loss)
        # 分类损失值除以(正例的个数加上使用的负例的个数)，得到分类损失的平均值
        total_loss /= (num_pos + num_neg_batch)
        # num_pos为所有正例的数量
        # tf.not_equal(num_pos, 0)为找到所有没有正例的batch，赋值为1
        # 这样做的原因是因为下边要求平均进行相除(batch,)
        num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos, tf.ones_like(num_pos))
        pos_loc_loss = tf.to_float(pos_loc_loss)
        # 总损失加上定位损失
        # 定位损失先乘以系数alpha再除以正例的个数
        total_loss += (self.alpha * pos_loc_loss) / num_pos
        assert isinstance(total_loss, object)
        return total_loss


if __name__ == '__main__':
    import numpy as np
    y_true = np.zeros((10,8732,33), dtype='double')
    y_true[:,[1,2,3,4,5],-8] = 1
    y_pred = np.zeros((10,8732,33))
    y_pred[:,:,22] += 1
    y_pred = tf.constant(y_pred)
    loss = MultiboxLoss(21)
    loss.compute_loss(y_true, y_pred)
