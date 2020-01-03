#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: a7_train.py
@Time: 2020-01-03 13:56
@Last_update: 2020-01-03 13:56
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import keras
import pickle
import tensorflow as tf
from a1_ssd300_model import SSD300Backbone
from a2_get_loc_conf import GetLocConf
from a4_bbox import BBoxUtility
from a5_generator import Generator
from a6_losses import MultiboxLoss


NUM_CLASSES = 21
input_shape = (300,300,3)


priors = pickle.load(open('data/prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

gt = pickle.load(open('data/VOC2007.p', 'rb'))
keys = sorted(gt.keys())
num_train = int(round(0.8 * len(keys)))
train_keys = keys[:num_train]
val_keys = keys[num_train:]
num_val = len(val_keys)

path_prefix = '/media/freshield/SSD_1T/Data/a13_detection/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'
gen = Generator(gt, bbox_util, 16, path_prefix,train_keys, val_keys, (input_shape[0], input_shape[1]), do_crop=False)

net = SSD300Backbone(input_shape)
model, net = GetLocConf(net, (input_shape[0], input_shape[1]), num_classes=NUM_CLASSES)

base_lr = 3e-4
def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

callbacks = [keras.callbacks.ModelCheckpoint('data/train/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=1,
                                             save_weights_only=True),
             keras.callbacks.LearningRateScheduler(schedule)]
optim = keras.optimizers.Adam(lr=base_lr)

model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)

config = tf.ConfigProto(allow_soft_placement=True)
# 开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
nb_epoch = 300
history = model.fit_generator(gen.generate(True), gen.train_batches,
                              nb_epoch, verbose=1,
                              callbacks=callbacks,
                              validation_data=gen.generate(False),
                              nb_val_samples=gen.val_batches,
                              nb_worker=1)