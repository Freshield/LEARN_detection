#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: t4_try_numpy_max.py
@Time: 2020-01-05 16:50
@Last_update: 2020-01-05 16:50
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

a = np.arange(12).reshape((3,4))
a[1,1] = 100
print(a)
print(a.max(axis=0))
print(a.argmax(axis=0))