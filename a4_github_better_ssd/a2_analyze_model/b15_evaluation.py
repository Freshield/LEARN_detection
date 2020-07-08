#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b15_evaluation.py
@Time: 2020-04-09 18:06
@Last_update: 2020-04-09 18:06
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
from keras.preprocessing import image
from imageio import imread
import os
import numpy as np
from math import ceil
import tensorflow as tf
import keras.backend as K
from keras.optimizers import SGD
from keras.callbacks import CSVLogger, LearningRateScheduler, TerminateOnNaN, ModelCheckpoint
from b5_ssd300_model import ssd300
from b7_SSDLoss import SSDLoss
from object_detection_2d_geometric_ops import Resize
from object_detection_2d_photometric_ops import ConvertTo3Channels
from b6_data_generator import DataGenerator
from b6_m1_data_augment import SSDDataAugmentation
from b6_m4_input_encoder import SSDInputEncoder
import matplotlib.pyplot as plt
from b9_decoder import decode_detections
from b10_apply_inverse_transforms import apply_inverse_transforms
from b15_m1_evaluator import Evaluator

# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.
train_hdf5_path = '../data/image_data/dataset_pascal_voc_07+12_trainval.h5'
val_hdf5_path = '../data/image_data/dataset_pascal_voc_07_test.h5'
batch_size = 8
img_height = 300  # Height of the model input images
img_width = 300  # Width of the model input images
img_channels = 3  # Number of color channels of the model input images
mean_color = [123, 117,
              104]  # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1,
                 0]  # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 20  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
                 1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87,
               1.05]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100,
         300]  # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
           0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2,
             0.2]  # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True

# 1: Build the Keras model.

K.clear_session() # Clear previous models from memory.

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

config = tf.ConfigProto(allow_soft_placement=True)
# 开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


model = ssd300(image_size=(img_height, img_width, img_channels),
               n_classes=n_classes,
               mode='training',
               l2_regularization=0.0005,
               scales=scales,
               aspect_ratios_per_layer=aspect_ratios,
               two_boxes_for_ar1=two_boxes_for_ar1,
               steps=steps,
               offsets=offsets,
               clip_boxes=clip_boxes,
               variances=variances,
               normalize_coords=normalize_coords,
               subtract_mean=mean_color,
               swap_channels=swap_channels)

# 2: Load some weights into the model.

# TODO: Set the path to the weights you want to load.
weights_path = '../data/weights/VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.h5'

model.load_weights(weights_path, by_name=True)

# 3: Instantiate an optimizer and the SSD loss function and compile the model.
#    If you want to follow the original Caffe implementation, use the preset SGD
#    optimizer, otherwise I'd recommend the commented-out Adam optimizer.

#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

val_dataset = DataGenerator(load_images_into_memory = False, hdf5_dataset_path = val_hdf5_path)
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)
predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'filenames',
                                                  'inverse_transform',
                                                  'original_images',
                                                  'original_labels'},
                                         keep_images_without_gt=False)

evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=val_dataset,
                      model_mode='training')

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=8,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='integrate',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results

print(average_precisions)
print()
print(mean_average_precision)