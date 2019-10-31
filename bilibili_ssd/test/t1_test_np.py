#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: t1_test_np.py
@Time: 2019-10-31 09:40
@Last_update: 2019-10-31 09:40
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
from scipy.misc import imread

y, x = np.mgrid[0:38, 0:38]
print(y)
print(x)

y = (y + 0.5) * 8 / 300
print(y * 300)
print(y)

print(37.5 * 8)
print(18.5 * 16)