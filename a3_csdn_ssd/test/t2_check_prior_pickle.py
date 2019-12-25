#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: t2_check_prior_pickle.py
@Time: 2019-11-01 19:43
@Last_update: 2019-11-01 19:43
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
pickle_path = '../data/prior_boxes_ssd300.pkl'

priors = pickle.load(open(pickle_path, 'rb'))
print(priors.shape)
print(priors[0])