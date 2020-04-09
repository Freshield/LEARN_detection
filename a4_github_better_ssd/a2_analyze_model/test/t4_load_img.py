#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: t4_load_img.py
@Time: 2020-04-01 16:33
@Last_update: 2020-04-01 16:33
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import cv2

image = cv2.imread('../../data/testssd_rst.png')
print(image.shape)
image = image[:700, 100:1900, :]

cv2.imshow('image', image)
cv2.waitKey()
