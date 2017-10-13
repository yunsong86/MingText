#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: observation_data.py
@time: 17-10-9 上午10:38
@desc: python3.6
"""

import os
import collections
from matplotlib import mlab
from  matplotlib import rcParams
import matplotlib.pyplot as plt


# 每个类别样本数量
def class_distribution(corpus_path):
    class_count_dit = collections.defaultdict(int)
    for class_file in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, class_file)
        for file in os.listdir(file_path):
            class_count_dit[class_file] += 1
    class_count_dit = sorted(class_count_dit.items(), key=lambda x: x[1])
    return class_count_dit

def



if __name__ == '__main__':
    corpus_path = '/mnt/hgfs/UbunutWin/corpus/news_data'
    class_count_dit = class_distribution(corpus_path)
