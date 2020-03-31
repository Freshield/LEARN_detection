#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b3_m1_anchorboxes.py
@Time: 2020-02-06 14:24
@Last_update: 2020-02-06 14:24
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
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer

from b3_m2_conver_coor import convert_coordinates

class AnchorBoxes(Layer):
    '''
    锚点先验框层
    A Keras layer to create an output tensor containing anchor box coordinates
    and variances based on the input tensor and the passed arguments.

    A set of 2D anchor boxes of different aspect ratios is created for each spatial unit of
    the input tensor. The number of anchor boxes created per unit depends on the arguments
    `aspect_ratios` and `two_boxes_for_ar1`, in the default case it is 4. The boxes
    are parameterized by the coordinate tuple `(xmin, xmax, ymin, ymax)`.

    The logic implemented by this layer is identical to the logic in the module
    `ssd_box_encode_decode_utils.py`.

    The purpose of having this layer in the network is to make the model self-sufficient
    at inference time. Since the model is predicting offsets to the anchor boxes
    (rather than predicting absolute box coordinates directly), one needs to know the anchor
    box coordinates in order to construct the final prediction boxes from the predicted offsets.
    If the model's output tensor did not contain the anchor box coordinates, the necessary
    information to convert the predicted offsets back to absolute coordinates would be missing
    in the model output. The reason why it is necessary to predict offsets to the anchor boxes
    rather than to predict absolute box coordinates directly is explained in `README.md`.

    Input shape:
        4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
        or `(batch, height, width, channels)` if `dim_ordering = 'tf'`.

    Output shape:
        5D tensor of shape `(batch, height, width, n_boxes, 8)`. The last axis contains
        the four anchor box coordinates and the four variance values for each box.
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 this_scale,
                 next_scale,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 two_boxes_for_ar1=True,
                 this_steps=None,
                 this_offsets=None,
                 clip_boxes=False,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 coords='centroids',
                 normalize_coords=False,
                 **kwargs):
        '''
        All arguments need to be set to the same values as in the box encoding process, otherwise the behavior is undefined.
        Some of these arguments are explained in more detail in the documentation of the `SSDBoxEncoder` class.

        Arguments:
            # 图像的宽高
            img_height (int): The height of the input images.
            img_width (int): The width of the input images.
            # 当前层的缩放别率
            this_scale (float): A float in [0, 1], the scaling factor for the size of the generated anchor boxes
                as a fraction of the shorter side of the input image.
            # 下一层的缩放比率，主要是为了计算特殊的最小的长宽比的先验框大小
            next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.
            # 先验框长宽比列表
            aspect_ratios (list, optional): The list of aspect ratios for which default boxes are to be
                generated for this layer.
            # 是否生成特殊的最小长宽比的先验框
            two_boxes_for_ar1 (bool, optional): Only relevant if `aspect_ratios` contains 1.
                If `True`, two default boxes will be generated for aspect ratio 1. The first will be generated
                using the scaling factor for the respective layer, the second one will be generated using
                geometric mean of said scaling factor and next bigger scaling factor.
            # 是否裁剪边缘先验框到图像内
            clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within image boundaries.
            # 图像的偏差值列表
            variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
                its respective variance value.
            # 图像坐标的表示方法，是中心还是左上右下
            coords (str, optional): The box coordinate format to be used internally in the model (i.e. this is not the input format
                of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width, and height),
                'corners' for the format `(xmin, ymin, xmax,  ymax)`, or 'minmax' for the format `(xmin, xmax, ymin, ymax)`.
            # 是否正则化坐标到0,1之间
            normalize_coords (bool, optional): Set to `True` if the model uses relative instead of absolute coordinates,
                i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates.
        '''
        # 断言保证
        if K.backend() != 'tensorflow':
            raise TypeError("This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(K.backend()))
        if (this_scale < 0) or (next_scale < 0) or (this_scale > 1):
            raise ValueError("`this_scale` must be in [0, 1] and `next_scale` must be >0, but `this_scale` == {}, `next_scale` == {}".format(this_scale, next_scale))
        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        # 保存输入值
        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.coords = coords
        self.normalize_coords = normalize_coords

        # 计算有多少个检验框
        # Compute the number of boxes per cell
        if (1 in aspect_ratios) and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)

        super(AnchorBoxes, self).__init__(**kwargs)

    # 没有需要训练的参数，所以build只是调用父类build
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    # 真正的调用部分
    def call(self, x, mask=None):
        '''
        Return an anchor box tensor based on the shape of the input tensor.

        The logic implemented here is identical to the logic in the module `ssd_box_encode_decode_utils.py`.

        Note that this tensor does not participate in any graph computations at runtime. It is being created
        as a constant once during graph creation and is just being output along with the rest of the model output
        during runtime. Because of this, all logic is implemented as Numpy array operations and it is sufficient
        to convert the resulting Numpy array into a Keras tensor at the very end before outputting it.

        Arguments:
            x (tensor): 4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
                or `(batch, height, width, channels)` if `dim_ordering = 'tf'`. The input for this
                layer must be the output of the localization predictor layer.
        '''
        # 得到长宽中最小的那个
        # Compute box width and height for each aspect ratio
        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        # 300
        size = min(self.img_height, self.img_width)
        # 根据长宽比和缩放比计算先验框的真实长宽
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        for ar in self.aspect_ratios: # [1.0, 2.0, 0.5]
            # 如果是第一个ar
            if (ar == 1):
                # 长宽等于当前缩放比乘以原图的大小
                # Compute the regular anchor box for aspect ratio 1.
                # 0.1 * 300 = 30
                box_height = box_width = self.this_scale * size
                wh_list.append((box_width, box_height))
                # 如果要加上特殊的长宽比
                if self.two_boxes_for_ar1:
                    # 长宽等于本层缩放比乘以下一层缩放比，开方，再乘以原图的大小
                    # Compute one slightly larger version using the geometric mean of this scale value and the next.
                    # ((0.1 * 0.2) ** 0.5) * 300
                    box_height = box_width = np.sqrt(self.this_scale * self.next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                # 其他的长等于缩放比乘以原图大小除以长宽比的开方
                # 0.1 * 300 / (2 ** 0.5), 0.1 * 300 / (0.5 ** 0.5)
                box_height = self.this_scale * size / np.sqrt(ar)
                # 其他的宽等于缩放比乘以原图大小乘以长宽比的开方
                # 0.1 * 300 * (2 ** 0.5), 0.1 * 300 * (0.5 ** 0.5)
                box_width = self.this_scale * size * np.sqrt(ar)
                # 这样算下来所有先验框的面积是相等的
                wh_list.append((box_width, box_height))
        # [[30,30],[42,42],[42,21],[21,42]]
        wh_list = np.array(wh_list)
        print(self.this_scale * size)
        print(wh_list)
        print()

        # 得到loc的shape
        # 38,38,4*4
        # We need the shape of the input tensor
        if K.image_dim_ordering() == 'tf':
            batch_size, feature_map_height, feature_map_width, feature_map_channels = x._keras_shape
        else: # Not yet relevant since TensorFlow is the only supported backend right now, but it can't harm to have this in here for the future
            batch_size, feature_map_channels, feature_map_height, feature_map_width = x._keras_shape

        # 计算所有先验框的中心坐标
        # Compute the grid of box center points. They are identical for all aspect ratios.

        # 计算所有先验框的中间距离
        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        # 如果没有给出先验框的距离则用原图长宽除以当前层的长宽
        if (self.this_steps is None):
            step_height = self.img_height / feature_map_height
            step_width = self.img_width / feature_map_width
        # 否则直接用给出的长宽
        # step_height: 8, step_width: 8
        else:
            if isinstance(self.this_steps, (list, tuple)) and (len(self.this_steps) == 2):
                step_height = self.this_steps[0]
                step_width = self.this_steps[1]
            elif isinstance(self.this_steps, (int, float)):
                step_height = self.this_steps
                step_width = self.this_steps

        # 计算偏移比率，也就是第一个先验框的中心点按照这个比率想乘来进行偏移
        # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
        # 如果没给出则使用默认的0.5
        if (self.this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        # 如果给出则直接使用
        # offset_height: 0.5, offset_width: 0.5
        else:
            if isinstance(self.this_offsets, (list, tuple)) and (len(self.this_offsets) == 2):
                offset_height = self.this_offsets[0]
                offset_width = self.this_offsets[1]
            elif isinstance(self.this_offsets, (int, float)):
                offset_height = self.this_offsets
                offset_width = self.this_offsets

        # 现在计算所有先验框的中心点
        # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        # 计算所有y的中心点，第一个y值为高的步长乘以高的偏移值，8*0.5=4
        # 最后一个值为偏移的高加上剩下的长度 0.5 * 8 + (38-1) * 8 = (0.5 + 38 - 1) * 8 = 300
        # 一共是当前层大小个先验框，38个
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height, feature_map_height)
        # 计算所有x的中心点，同上
        # [4, 12, 20, 28, 36, 44 ... , 276, 284, 292, 300]
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width, feature_map_width)
        # 把x和y的坐标网格化，得到cx_grid和cy_grid都是38*38的矩阵，对应的是38*38个坐标，
        # 表示所有先验框的中心坐标
        # cx_grid: [[4,12,20...,292,300],[4,12,20...,292,300],...,[4,12,20...,292,300]]
        # cy_grid: [[4,4,4...,4,4], [12,12,12...,12,12],...,[292,292,292,...,292], [300,300,...,300]]
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        # 增加一个维度方便之后的层叠
        # cx_grid: (38,38,1), cy_grid: (38,38,1)
        cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down

        # 创建本层的预测框，(38,38,4,4)，最后四维是中心点坐标加上宽和高，(cx,cy,w,h)
        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

        # 先把cx,cy复制4次，然后存到相应的位置
        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes)) # Set cy
        # 把w,h存到相应的位置，这里wh_list只有4个值，会自动拓展为38,38,4
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h

        # 把(cx,cy,w,h)的所有坐标转换为左上右下坐标(xmin,xmax,ymin,ymax)
        # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # 如果要裁剪边缘则把越界的都设为边缘值
        # If `clip_boxes` is enabled, clip the coordinates to lie within the image boundaries
        if self.clip_boxes:
            x_coords = boxes_tensor[:,:,:,[0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:,:,:,[0, 2]] = x_coords
            y_coords = boxes_tensor[:,:,:,[1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:,:,:,[1, 3]] = y_coords

        # 如果要归一化坐标，则用相应的坐标除以原图的大小
        # If `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth.
        # 如果重要需要的是中心坐标则把左上右下的坐标转换回来
        if self.coords == 'centroids':
            # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)`.
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids', border_pixels='half')
        elif self.coords == 'minmax':
            # Convert `(xmin, ymin, xmax, ymax)` to `(xmin, xmax, ymin, ymax).
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax', border_pixels='half')

        # 添加坐标的偏差值
        # Create a tensor to contain the variances and append it to `boxes_tensor`. This tensor has the same shape
        # as `boxes_tensor` and simply contains the same 4 variance values for every position in the last axis.
        # 创建零矩阵38,38,4,4
        variances_tensor = np.zeros_like(boxes_tensor) # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # 加上偏差值，等于赋值
        variances_tensor += self.variances # Long live broadcasting
        # 把坐标值和偏差值垒起来
        # Now `boxes_tensor` becomes a tensor of shape `(feature_map_height, feature_map_width, n_boxes, 8)`
        # (38,38,4,8)
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        # 把先验框的坐标numpy矩阵转换为keras的tensor
        # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
        # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
        # 先拓展第零维，然后进行层叠，这样恢复batch的大小
        # (batch, 38, 38, 4, 8)
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))

        return boxes_tensor

    # 计算输出的大小
    def compute_output_shape(self, input_shape):
        if K.image_dim_ordering() == 'tf':
            batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        else: # Not yet relevant since TensorFlow is the only supported backend right now, but it can't harm to have this in here for the future
            batch_size, feature_map_channels, feature_map_height, feature_map_width = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)

    def get_config(self):
        config = {
            'img_height': self.img_height,
            'img_width': self.img_width,
            'this_scale': self.this_scale,
            'next_scale': self.next_scale,
            'aspect_ratios': list(self.aspect_ratios),
            'two_boxes_for_ar1': self.two_boxes_for_ar1,
            'clip_boxes': self.clip_boxes,
            'variances': list(self.variances),
            'coords': self.coords,
            'normalize_coords': self.normalize_coords
        }
        base_config = super(AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
