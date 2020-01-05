#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: bbox.py
@Time: 2019-12-26 11:50
@Last_update: 2019-12-26 11:50
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


class BBoxUtility(object):
    """用来计算预测框和先验框的类

    # 参数
        num_classes: 包括背景在内的类别数
        priors: 先验框位置和variance (num_priors, 8),
            priors[i] = [xmin, ymin, xmax, ymax, varxc, varyc, varw, varh].
        overlap_threshold: 先验框分配的阈值
        nms_thresh: NMS的阈值
        top_k: 每张图在NMS之后保留的最大张数

    # References
        https://arxiv.org/abs/1512.02325
    """
    # TODO add setter methods for nms_thresh and top_K
    def __init__(self, num_classes, priors=None, overlap_threshold=0.5,
                 nms_thresh=0.45, top_k=400):
        # 类别数 21
        self.num_classes = num_classes
        # 先验框 (8732, 4+4)
        self.priors = priors
        # 先验框个数 8732
        self.num_priors = 0 if priors is None else len(priors)
        # 先验框分配阈值 0.5
        self.overlap_threshold = overlap_threshold
        # NMS的阈值 0.45
        self._nms_thresh = nms_thresh
        # 每张图NMS之后保留的最大张数
        self._top_k = top_k
        # 预测框的位置信息
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        # 预测框的得分信息
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        # NMS算法
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

    # NMS的setter和getter
    @property
    def nms_thresh(self):
        return self._nms_thresh
    @nms_thresh.setter
    def nms_thresh(self, value):
        self._nms_thresh = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)
    # top_k的setter和getter
    @property
    def top_k(self):
        return self._top_k
    @top_k.setter
    def top_k(self, value):
        self._top_k = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)

    def iou(self, box):
        """
        把给出的box的位置和所有的先验框来计算IOU
        # 参数
            box: 预测框 (4,).

        # 返回
            iou: iou数值,
                numpy tensor of shape (num_priors), (8732).
        """
        # priors (8732, 4+4)
        # box (xmin,ymin,xmax,ymax)
        # 计算所有先验框和特定box的相交部分
        # 这里得到xmin, ymin的最大值
        inter_botleft = np.maximum(self.priors[:, :2], box[:2])
        # 这里得到xmax, ymax的最小值
        inter_upright = np.minimum(self.priors[:, 2:4], box[2:])
        # 得到(xmax-xmin, ymax-ymin),这里得到相交的面积
        # 相当于把面积问题变成了线段问题
        inter_wh = inter_upright - inter_botleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]

        # 计算相应先验框和特定box的总体面积部分
        # 计算box的面积
        area_pred = (box[2] - box[0]) * (box[3] - box[1])
        # 计算所有先验框的总体面积
        area_gt = (self.priors[:, 2] - self.priors[:, 0])
        area_gt *= (self.priors[:, 3] - self.priors[:, 1])
        # 计算IOU
        union = area_pred + area_gt - inter
        # 8732
        iou = inter / union

        return iou

    def encode_box(self, box, return_iou=True):
        """
        这里把先验框和目标box比较后IOU大于阈值的进行编码，
        返回虽然是所有8732，但是除了IOU大于阈值的其他均为0
        # 参数
            box: label框 (4,).
            return_iou: 是否把IOU和编码值一同返回.

        # 返回
            encoded_box: 编码后的先验框和label框 (num_priors * (4 + int(return_iou))) (8732 * 5)
        """
        # 得到所有label框和先验框计算后的IOU值
        iou = self.iou(box)
        # (8732, 4 or 5)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))
        # 得到所有IOU大于先验框分配阈值,结果是(8732),类型是bool
        assign_mask = iou > self.overlap_threshold
        # any()这里代表是对所有值取或,也就是只要有IOU大于先验框分配阈值这里就为True
        # 前边有not,所以这里是没有一个IOU大于分配阈值的情况
        if not assign_mask.any():
            # 如果所有IOU均小于先验框分配阈值,则把IOU最大的那项设为True
            assign_mask[iou.argmax()] = True
        # 如果要返回IOU,
        if return_iou:
            # 把IOU大于阈值的最后一项设为IOU的值
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        # 只把IOU大于阈值的先验框选出来,这里设为x个, assigned_priors (x, 8)
        assigned_priors = self.priors[assign_mask]
        # box (xmin,ymin,xmax,ymax)
        # 得到label框的中心,box_center (x_center, y_center)
        box_center = 0.5 * (box[:2] + box[2:])
        # 得到label框的长宽,box_wh (w, h)
        box_wh = box[2:] - box[:2]
        # 这里得到所有IOU大于阈值的先验框的中心, prior_center (x, 2)
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        # 这里得到所有IOU大于阈值的先验框的长宽, prior_wh (x, 2)
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])

        # 这里带着variance来进行编码
        # encoded_box的计算公式
        # lx = ((bx - dx) / w) / variance[0]
        # ly = ((by - dy) / h) / variance[1]
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] /= assigned_priors[:, -4:-2]
        # lw = (log(bw/dw) / variance[2]
        # lh = (log(bh/dh) / variance[3]
        encoded_box[:, 2:4][assign_mask] = np.log(box_wh /
                                                  assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] /= assigned_priors[:, -2:]
        # 最终这里ravel为把数值全部拉平, (8732*4 or 5)
        # 这里的最终结果虽然为8732,但是只有IOU大于阈值的才有值，其他的均为0
        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        # 变量：
        #   boxes: label框,这里的num_boxes为做完数据增广之后的数量,也可以粗略的理解为batch_size
        #   (num_boxes, 4 + num_classes),其中num_classes没有包括背景
        # 返回值：
        #   assignment：分配后的预测框(num_boxes, 4 + num_classes + 8),
        # 第二维上的8其实很多都是0，只有在assignment[:, -8]存在1，代表给default box分配了哪个groud truth
        # (8732, 4+21+8)
        assignment = np.zeros((self.num_priors, 4 + self.num_classes + 8))
        assignment[:, 4] = 1.0
        # 如果预测框数量为0,则直接返回
        if len(boxes) == 0:
            return assignment
        # 针对boxes的第1维的所有值都应用encode_box, encoded_boxes (num_boxes, 8732*5)
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        # reshape后, encoded_boxes (num_boxes, 8732, 5)
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)
        # 找出
        # 找出一张图中的所有的object与所有的prior box的最大IOU，即每个prior box对应一个object
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        ##找出每个prior box对应的那个object的索引。len(best_iou_idx)=num_priors
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        ##找出与groud truth 存在IOU的prior box
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]
        assign_num = len(best_iou_idx)
        ##筛选出与groud truth 有IOU的prior box
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        # 确定给assignment分配中的prior box分配 具体哪一个groud truth。best_iou_idx中元素的范围为：range(num_object)。
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -8][best_iou_mask] = 1
        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox, variances):
        """Convert bboxes from local predictions to shifted priors.

        # Arguments
            mbox_loc: Numpy array of predicted locations.
            mbox_priorbox: Numpy array of prior boxes.
            variances: Numpy array of variances.

        # Return
            decode_bbox: Shifted priors.
        """
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0]
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_width * variances[:, 1]
        decode_bbox_center_y += prior_center_y
        decode_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
        decode_bbox_height *= prior_height
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(self, predictions, background_label_id=0, keep_top_k=200,
                      confidence_threshold=0.01):
        """Do non maximum suppression (nms) on prediction results.

        # Arguments
            predictions: Numpy array of predicted values.
            num_classes: Number of classes for prediction.
            background_label_id: Label of background class.
            keep_top_k: Number of total bboxes to be kept per image
                after nms step.
            confidence_threshold: Only consider detections,
                whose confidences are larger than a threshold.

        # Return
            results: List of predictions for every picture. Each prediction is:
                [label, confidence, xmin, ymin, xmax, ymax]
        """
        mbox_loc = predictions[:, :, :4]
        variances = predictions[:, :, -4:]
        mbox_priorbox = predictions[:, :, -8:-4]
        mbox_conf = predictions[:, :, 4:-8]
        results = []
        for i in range(len(mbox_loc)):
            results.append([])
            decode_bbox = self.decode_boxes(mbox_loc[i],
                                            mbox_priorbox[i], variances[i])
            for c in range(self.num_classes):
                if c == background_label_id:
                    continue
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence_threshold
                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]
                    feed_dict = {self.boxes: boxes_to_process,
                                 self.scores: confs_to_process}
                    idx = self.sess.run(self.nms, feed_dict=feed_dict)
                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]
                    labels = c * np.ones((len(idx), 1))
                    c_pred = np.concatenate((labels, confs, good_boxes),
                                            axis=1)
                    results[-1].extend(c_pred)
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                argsort = np.argsort(results[-1][:, 1])[::-1]
                results[-1] = results[-1][argsort]
                results[-1] = results[-1][:keep_top_k]
        return results


if __name__ == '__main__':
    import pickle
    priors = pickle.load(open('data/prior_boxes_ssd300.pkl', 'rb'))
    bbox_util = BBoxUtility(21, priors)
    test_box = np.array([0.2,0.2,0.5,0.5])
    bbox_util.encode_box(test_box)
