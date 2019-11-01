#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: t1_test_l2norm.py
@Time: 2019-10-31 17:02
@Last_update: 2019-10-31 17:02
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
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

a = np.arange(12, dtype=np.float32).reshape((3,4))
print(a)
b = tf.keras.backend.l2_normalize(a)
print(b)
c = a / np.sqrt(np.sum(a**2))
print(c)