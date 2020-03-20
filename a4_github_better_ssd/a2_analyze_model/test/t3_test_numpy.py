#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: t3_test_numpy.py
@Time: 2020-03-20 17:27
@Last_update: 2020-03-20 17:27
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

pred = np.array([3,4,5,6,7,1,2,3,4,5,6,7,8,9,10])
top_k = 5

print(pred)
print(len(pred)-top_k)
part = np.argpartition(pred, kth=len(pred)-top_k, axis=0)
print(part)
print(pred[part])
print()
select = part[len(pred)-top_k:]
print(select)
rst = pred[select]
print(rst)
