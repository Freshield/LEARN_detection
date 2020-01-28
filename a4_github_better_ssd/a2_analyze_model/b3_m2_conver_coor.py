#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b3_m2_conver_coor.py
@Time: 2020-01-29 16:42
@Last_update: 2020-01-29 16:42
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

# 进行坐标的转换
def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):
    '''
    Convert coordinates for axis-aligned 2D boxes between two coordinate formats.

    Creates a copy of `tensor`, i.e. does not operate in place. Currently there are
    three supported coordinate formats that can be converted from and to each other:
        1) (xmin, xmax, ymin, ymax) - the 'minmax' format
        2) (xmin, ymin, xmax, ymax) - the 'corners' format
        2) (cx, cy, w, h) - the 'centroids' format

    Arguments:
        # 输入的numpy矩阵
        tensor (array): A Numpy nD array containing the four consecutive coordinates
            to be converted somewhere in the last axis.
        # 坐标的开始索引
        start_index (int): The index of the first coordinate in the last axis of `tensor`.
        # 转换模式
        conversion (str, optional): The conversion direction. Can be 'minmax2centroids',
            'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners',
            or 'corners2minmax'.
        # 边缘模式如何处理
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A Numpy nD array, a copy of the input tensor with the converted coordinates
        in place of the original coordinates and the unaltered elements of the original
        tensor elsewhere.
    '''
    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+1]) / 2.0 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+2] + tensor[..., ind+3]) / 2.0 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind] + d # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+2] + d # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
    # 转换左上右下到中心
    elif conversion == 'corners2centroids':
        # cx = (xmin + xmax) / 2
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+2]) / 2.0 # Set cx
        # cy = (ymin + ymax) / 2
        tensor1[..., ind+1] = (tensor[..., ind+1] + tensor[..., ind+3]) / 2.0 # Set cy
        # w = xmax - xmin
        tensor1[..., ind+2] = tensor[..., ind+2] - tensor[..., ind] + d # Set w
        # h = ymax - ymin
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+1] + d # Set h
    # 把中心坐标转换为左上右下坐标
    elif conversion == 'centroids2corners':
        # xmin = cx - (w / 2)
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        # ymin = cy - (h / 2)
        tensor1[..., ind+1] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        # xmax = cx + (w / 2)
        tensor1[..., ind+2] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        # ymax = cy + (h / 2)
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
    elif (conversion == 'minmax2corners') or (conversion == 'corners2minmax'):
        tensor1[..., ind+1] = tensor[..., ind+2]
        tensor1[..., ind+2] = tensor[..., ind+1]
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids', 'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners', and 'corners2minmax'.")

    return tensor1

