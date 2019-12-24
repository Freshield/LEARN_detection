#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b13_train_512.py
@Time: 2020-03-26 17:09
@Last_update: 2020-03-26 17:09
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
from keras.optimizers import Adam
from b12_ssd512 import ssd_512
from math import ceil
import keras.backend as K
from keras.callbacks import CSVLogger, LearningRateScheduler, TerminateOnNaN, ModelCheckpoint
from b7_SSDLoss import SSDLoss
from object_detection_2d_geometric_ops import Resize
from object_detection_2d_photometric_ops import ConvertTo3Channels
from b6_data_generator import DataGenerator
from b6_m1_data_augment import SSDDataAugmentation
from b6_m4_input_encoder import SSDInputEncoder


import tensorflow as tf

train_hdf5_path = '../data/image_data/dataset_pascal_voc_07+12_trainval.h5'
val_hdf5_path = '../data/image_data/dataset_pascal_voc_07_test.h5'
img_height = 512 # Height of the model input images
img_width = 512 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 20 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_pascal
aspect_ratios=[[1.0, 2.0, 0.5],
               [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
               [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
               [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
               [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
               [1.0, 2.0, 0.5],
               [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 128, 256, 512] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True

# 1: Build the Keras model.

K.clear_session() # Clear previous models from memory.

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

config = tf.ConfigProto(allow_soft_placement=True)
# 开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


model = ssd_512(image_size=(img_height, img_width, 3),
                n_classes=20,
                mode='training',
                l2_regularization=0.0005,
                scales=scales, # The scales for MS COCO are [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06]
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=[2, 1, 0])

# 2: Load some weights into the model.

# TODO: Set the path to the weights you want to load.
weights_path = '../data/weights/VGG_ILSVRC_16_layers_fc_reduced.h5'

model.load_weights(weights_path, by_name=True)

# 3: Instantiate an optimizer and the SSD loss function and compile the model.
#    If you want to follow the original Caffe implementation, use the preset SGD
#    optimizer, otherwise I'd recommend the commented-out Adam optimizer.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.

train_dataset = DataGenerator(load_images_into_memory = False, hdf5_dataset_path = train_hdf5_path)
val_dataset = DataGenerator(load_images_into_memory = False, hdf5_dataset_path = val_hdf5_path)

# 2: Parse the image and label lists for the training and validation datasets. This can take a while.

# TODO: Set the paths to the datasets here.

# # The directories that contain the images.
# VOC_2007_images_dir      = '/media/freshield/SSD_1T/Data/a13_detection/VOCdevkit/VOC2007/JPEGImages/'
# VOC_2012_images_dir      = '/media/freshield/SSD_1T/Data/a13_detection/VOCdevkit/VOC2012/JPEGImages/'
#
# # The directories that contain the annotations.
# VOC_2007_annotations_dir      = '/media/freshield/SSD_1T/Data/a13_detection/VOCdevkit/VOC2007/Annotations/'
# VOC_2012_annotations_dir      = '/media/freshield/SSD_1T/Data/a13_detection/VOCdevkit/VOC2012/Annotations/'
#
# # The paths to the image sets.
# VOC_2007_train_image_set_filename    = '/media/freshield/SSD_1T/Data/a13_detection/VOCdevkit/VOC2007/ImageSets/Main/train.txt'
# VOC_2012_train_image_set_filename    = '/media/freshield/SSD_1T/Data/a13_detection/VOCdevkit/VOC2012/ImageSets/Main/train.txt'
# VOC_2007_val_image_set_filename      = '/media/freshield/SSD_1T/Data/a13_detection/VOCdevkit/VOC2007/ImageSets/Main/val.txt'
# VOC_2012_val_image_set_filename      = '/media/freshield/SSD_1T/Data/a13_detection/VOCdevkit/VOC2012/ImageSets/Main/val.txt'
# VOC_2007_trainval_image_set_filename = '/media/freshield/SSD_1T/Data/a13_detection/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
# VOC_2012_trainval_image_set_filename = '/media/freshield/SSD_1T/Data/a13_detection/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
# VOC_2007_test_image_set_filename     = '/media/freshield/SSD_1T/Data/a13_detection/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

# train_dataset.parse_xml(images_dirs=[VOC_2007_images_dir,
#                                      VOC_2012_images_dir],
#                         image_set_filenames=[VOC_2007_trainval_image_set_filename,
#                                              VOC_2012_trainval_image_set_filename],
#                         annotations_dirs=[VOC_2007_annotations_dir,
#                                           VOC_2012_annotations_dir],
#                         classes=classes,
#                         include_classes='all',
#                         exclude_truncated=False,
#                         exclude_difficult=False,
#                         ret=False)
#
# val_dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
#                       image_set_filenames=[VOC_2007_test_image_set_filename],
#                       annotations_dirs=[VOC_2007_annotations_dir],
#                       classes=classes,
#                       include_classes='all',
#                       exclude_truncated=False,
#                       exclude_difficult=True,
#                       ret=False)

# Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
# speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
# option in the constructor, because in that cas the images are in memory already anyway. If you don't
# want to create HDF5 datasets, comment out the subsequent two function calls.

# train_dataset.create_hdf5_dataset(file_path='data/image_data/dataset_pascal_voc_07+12_trainval.h5',
#                                   resize=False,
#                                   variable_image_size=True,
#                                   verbose=True)

# val_dataset.create_hdf5_dataset(file_path='data/image_data/dataset_pascal_voc_07_test.h5',
#                                 resize=False,
#                                 variable_image_size=True,
#                                 verbose=True)
# train_dataset.hdf5_dataset = h5py.File('data/image_data/dataset_pascal_voc_07+12_trainval.h5', 'r')
# train_dataset.hdf5_dataset_path = 'data/image_data/dataset_pascal_voc_07+12_trainval.h5'
# train_dataset.dataset_size = len(train_dataset.hdf5_dataset['images'])
# train_dataset.dataset_indices = np.arange(train_dataset.dataset_size, dtype=np.int32)
#
#
# val_dataset.hdf5_dataset = h5py.File('data/image_data/dataset_pascal_voc_07_test.h5', 'r')
# val_dataset.hdf5_dataset_path = 'data/image_data/dataset_pascal_voc_07_test.h5'
# val_dataset.dataset_size = len(val_dataset.hdf5_dataset['images'])
# val_dataset.dataset_indices = np.arange(val_dataset.dataset_size, dtype=np.int32)

# 3: Set the batch size.

batch_size = 8 # Change the batch size if you like, or if you run into GPU memory issues.

# 4: Set the image transformations for pre-processing and data augmentation options.

# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv10_2_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# Define a learning rate schedule.

def lr_schedule(epoch):
    if epoch <=1:
        return 0.0001
    elif epoch < 80:
        return 0.0005
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001


# Define model callbacks.

# TODO: Set the filepath under which you want to save the model.
model_checkpoint = ModelCheckpoint(filepath='../data/weights/example_trained/ssd512_pascal_07+12_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)
#model_checkpoint.best =

csv_logger = CSVLogger(filename='../data/ssd512_pascal_07+12_training_log.csv',
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch   = 0
final_epoch     = 120
steps_per_epoch = 1000


history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)