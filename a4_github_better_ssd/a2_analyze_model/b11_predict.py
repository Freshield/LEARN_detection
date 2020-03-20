#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b9_predict.py
@Time: 2020-03-20 10:46
@Last_update: 2020-03-20 10:46
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
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
weights_path = '../data/weights/example_trained/ssd300_pascal_07+12_epoch-53_loss-4.4248_val_loss-4.3018.h5'

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

batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)

i = 0 # Which batch item to look at

print("Image:", batch_images.shape)
print()
print("Ground truth boxes:\n")
print(np.array(batch_original_labels[i]))

y_pred = model.predict(batch_images)

# 4: Decode the raw predictions in `y_pred`.
# 得到decode之后的值，(n_objects,6)，分类号+预测的分类值+xmin,ymin,xmax,ymax
y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.4,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)

# 5: Convert the predictions for the original image.
# 使用反预处理，这里主要就是一个反resize
# 这里就是计算resize回去的位置
# labels[:, [ymin+1, ymax+1]] = np.round(labels[:, [ymin+1, ymax+1]] * (img_height / self.out_height), decimals=0)
# labels[:, [xmin+1, xmax+1]] = np.round(labels[:, [xmin+1, xmax+1]] * (img_width / self.out_width), decimals=0)
y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)


np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded_inv[i])

# 5: Draw the predicted boxes onto the image
# 观看预测的结果
# Set the colors for the bounding boxes
# 设置不同的类别不同的颜色
colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
# 设置背景类
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

plt.figure(figsize=(20,12))
plt.imshow(batch_original_images[i])

current_axis = plt.gca()
# 遍历label的框
for box in batch_original_labels[i]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    label = '{}'.format(classes[int(box[0])])
    # 画检测框
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
    # 写label的名称
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

# 遍历预测的框
for box in y_pred_decoded_inv[i]:
    xmin = box[2]
    ymin = box[3]
    xmax = box[4]
    ymax = box[5]
    # 得到相应类别的颜色
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    # 画检验框
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
    # 写label的名称
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

plt.show()
plt.close()