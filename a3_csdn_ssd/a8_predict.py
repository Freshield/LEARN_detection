#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: a8_predict.py
@Time: 2020-01-17 10:11
@Last_update: 2020-01-17 10:11
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import pickle
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from imageio import imread
from skimage.transform import resize as imresize
from keras.applications.imagenet_utils import preprocess_input
from a1_ssd300_model import SSD300Backbone
from a2_get_loc_conf import GetLocConf
from a4_bbox import BBoxUtility
import matplotlib.pyplot as plt

NUM_CLASSES = 21
input_shape = (300,300,3)
path_prefix = '/media/freshield/SSD_1T/Data/a13_detection/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/'
gt = pickle.load(open('data/VOC2007.p', 'rb'))
keys = sorted(gt.keys())
num_train = int(round(0.8 * len(keys)))
train_keys = keys[:num_train]
val_keys = keys[num_train:]
num_val = len(val_keys)
priors = pickle.load(open('data/prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

config = tf.ConfigProto(allow_soft_placement=True)
# 开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

inputs = []
images = []
img_path = path_prefix + sorted(val_keys)[0]
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
inputs = preprocess_input(np.array(inputs))
net = SSD300Backbone(input_shape)
model, net = GetLocConf(net, (input_shape[0], input_shape[1]), num_classes=NUM_CLASSES)
weights_path = 'data/weights.hdf5'
model.load_weights(weights_path)

preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)

for i, img in enumerate(images):
    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()

    plt.imshow(img / 255.)
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        #         label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label)
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

    plt.show()