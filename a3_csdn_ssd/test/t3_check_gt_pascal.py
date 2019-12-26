#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: t3_check_gt_pascal.py
@Time: 2019-12-26 20:20
@Last_update: 2019-12-26 20:20
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
data_path = '../data/gt_pascal.pkl'
with open(data_path, 'rb') as f:
    gt_data = pickle.loads(f.read())

print(gt_data)
print(len(gt_data))