#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: com_resources.py
@time: 17-9-28 上午3:52
@desc: python3.6
"""

STOP_WORDS_DIR = '/mnt/hgfs/UbunutWin/resources/stopwords.txt'
file_read = open(STOP_WORDS_DIR, 'r', encoding='utf-8')
STOP_WORDS_LIST = [w.strip() for w in file_read if w]
file_read.close()
