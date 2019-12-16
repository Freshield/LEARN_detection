#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: t2_test_numpy.py
@Time: 2020-03-18 15:04
@Last_update: 2020-03-18 15:04
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


a = np.zeros((5,4,3))
print(a[...,[0,1]].shape)