#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b6_m5_matching_utils.py
@Time: 2020-03-19 13:56
@Last_update: 2020-03-19 13:56
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

def match_bipartite_greedy(weight_matrix):
    '''
    贪心二分匹配
    相当于找出所有label和所有先验框中最匹配的那个，然后把这个label和相应的先验框去除后
    再找出剩下的label和剩下的先验框最匹配的那个，以此循环找出所有label和对应的先验框
    Returns a bipartite matching according to the given weight matrix.

    The algorithm works as follows:

    Let the first axis of `weight_matrix` represent ground truth boxes
    and the second axis anchor boxes.
    The ground truth box that has the greatest similarity with any
    anchor box will be matched first, then out of the remaining ground
    truth boxes, the ground truth box that has the greatest similarity
    with any of the remaining anchor boxes will be matched second, and
    so on. That is, the ground truth boxes will be matched in descending
    order by maximum similarity with any of the respectively remaining
    anchor boxes.
    The runtime complexity is O(m^2 * n), where `m` is the number of
    ground truth boxes and `n` is the number of anchor boxes.

    Arguments:
        weight_matrix (array): A 2D Numpy array that represents the weight matrix
            for the matching process. If `(m,n)` is the shape of the weight matrix,
            it must be `m <= n`. The weights can be integers or floating point
            numbers. The matching process will maximize, i.e. larger weights are
            preferred over smaller weights.

    Returns:
        A 1D Numpy array of length `weight_matrix.shape[0]` that represents
        the matched index along the second axis of `weight_matrix` for each index
        along the first axis.
    '''
    # 首先复制iou矩阵
    weight_matrix = np.copy(weight_matrix) # We'll modify this array.
    # n_objects
    num_ground_truth_boxes = weight_matrix.shape[0]
    # 生成索引矩阵列表，(n_objects,)
    all_gt_indices = list(range(num_ground_truth_boxes)) # Only relevant for fancy-indexing below.

    # This 1D array will contain for each ground truth box the index of
    # the matched anchor box.
    # 生成numpy矩阵，(n_objects,)
    matches = np.zeros(num_ground_truth_boxes, dtype=np.int)

    # In each iteration of the loop below, exactly one ground truth box
    # will be matched to one anchor box.
    # 遍历n_objects次
    for _ in range(num_ground_truth_boxes):
        # Find the maximal anchor-ground truth pair in two steps: First, reduce
        # over the anchor boxes and then reduce over the ground truth boxes.
        # 首先找到各个label的最大值索引
        anchor_indices = np.argmax(weight_matrix, axis=1) # Reduce along the anchor box axis.
        # 找出各个label的最大值
        overlaps = weight_matrix[all_gt_indices, anchor_indices]
        # 得到iou值最大的label索引
        ground_truth_index = np.argmax(overlaps) # Reduce along the ground truth box axis.
        # 得到iou值最大的label的最大值索引
        anchor_index = anchor_indices[ground_truth_index]
        # matches的iou最大的label的索引位置填上相应的索引值
        matches[ground_truth_index] = anchor_index # Set the match.

        # Set the row of the matched ground truth box and the column of the matched
        # anchor box to all zeros. This ensures that those boxes will not be matched again,
        # because they will never be the best matches for any other boxes.
        # 把已经匹配完的label和已经匹配完的先验框都归零
        weight_matrix[ground_truth_index] = 0
        weight_matrix[:,anchor_index] = 0

    return matches

def match_multi(weight_matrix, threshold):
    '''
    匹配所有iou大于阈值的先验框
    Matches all elements along the second axis of `weight_matrix` to their best
    matches along the first axis subject to the constraint that the weight of a match
    must be greater than or equal to `threshold` in order to produce a match.

    If the weight matrix contains elements that should be ignored, the row or column
    representing the respective elemet should be set to a value below `threshold`.

    Arguments:
        weight_matrix (array): A 2D Numpy array that represents the weight matrix
            for the matching process. If `(m,n)` is the shape of the weight matrix,
            it must be `m <= n`. The weights can be integers or floating point
            numbers. The matching process will maximize, i.e. larger weights are
            preferred over smaller weights.
        threshold (float): A float that represents the threshold (i.e. lower bound)
            that must be met by a pair of elements to produce a match.

    Returns:
        Two 1D Numpy arrays of equal length that represent the matched indices. The first
        array contains the indices along the first axis of `weight_matrix`, the second array
        contains the indices along the second axis.
    '''
    # 先验框的数量，8732
    num_anchor_boxes = weight_matrix.shape[1]
    # 得到先验框的索引列表，0,1,2...,8731
    all_anchor_indices = list(range(num_anchor_boxes)) # Only relevant for fancy-indexing below.

    # Find the best ground truth match for every anchor box.
    # 找到每个先验框最匹配的label，(8732,)
    ground_truth_indices = np.argmax(weight_matrix, axis=0) # Array of shape (weight_matrix.shape[1],)
    # 得到最匹配的iou值
    overlaps = weight_matrix[ground_truth_indices, all_anchor_indices] # Array of shape (weight_matrix.shape[1],)

    # Filter out the matches with a weight below the threshold.
    # 得到所有iou大于阈值的索引矩阵
    anchor_indices_thresh_met = np.nonzero(overlaps >= threshold)[0]
    # 得到所有iou大于阈值的先验框的匹配的label索引
    gt_indices_thresh_met = ground_truth_indices[anchor_indices_thresh_met]

    return gt_indices_thresh_met, anchor_indices_thresh_met


if __name__ == '__main__':
    print(np.nonzero([True,True,False,False,True,False]))