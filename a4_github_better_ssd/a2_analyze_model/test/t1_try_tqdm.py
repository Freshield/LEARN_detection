# coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: t1_try_tqdm.py
@Time: 2020-02-09 20:26
@Last_update: 2020-02-09 20:26
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
from tqdm import tqdm
from time import sleep

it = tqdm(range(100), 'Processing num')

for i in it:
    sleep(0.1)