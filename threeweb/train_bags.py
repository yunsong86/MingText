#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@license: (C) Copyright 2013-2017,Bertadata Corporation Limited.
@contact: yunsong@bertadata.com
@file: train_bags.py
@time: 2017/9/10 17:05
@desc: python2
"""
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import sys
import os
# 引入Bunch类
from sklearn.datasets.base import Bunch
# 引入持久化类
import pickle
import jieba
from sklearn.feature_extraction.text import HashingVectorizer

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

# 分词后分类语料库路径
corpus_path = "E:/NLP_DATA_BERTA/corpus/threeweb/text_mining/text_corpus_segment" + "/"
# 词袋语料路径
wordbag_path = "E:/NLP_DATA_BERTA/corpus/threeweb/text_mining/text_corpus_wordbag" + "/"

# data_set
# Bunch类提供一种key,value的对象形式
# target_name:所有分类集名称列表
# label:每个文件的分类标签列表
# filenames:文件名称
# contents:文件内容
data_set = Bunch(target_name=[], label=[], filenames=[], contents=[])

# 获取corpus_path下的所有子分类
dir_list = os.listdir(corpus_path)
data_set.target_name = dir_list
# 获取每个目录下所有的文件
for mydir in dir_list:
    class_path = corpus_path + mydir + "/"  # 拼出分类子目录的路径
    file_list = os.listdir(class_path)  # 获取class_path下的所有文件
    for file_path in file_list:  # 遍历所有文档
        file_name = class_path + file_path  # 拼出文件名全路径
        data_set.filenames.append(file_name)  # 把文件路径附加到数据集中
        data_set.label.append(data_set.target_name.index(mydir))  # 把文件分类标签附加到数据集中
        file_read = open(file_name, 'rb')  # 打开一个文件
        seg_corpus = file_read.read()  # 读取语料
        data_set.contents.append(seg_corpus)  # 构建分词文本内容列表
        file_read.close()

# 训练集对象持久化
file_obj = open(wordbag_path + "train_set.data", "wb")
pickle.dump(data_set, file_obj)
file_obj.close()
file_obj = open(wordbag_path + "train_set.data", "rb")
data_set = {}  # 清空原有数据集

# 验证持久化结果：
# 读取持久化后的对象
data_set = pickle.load(file_obj)
file_obj.close()
# 输出数据集包含的所有类别
print data_set.target_name
# 输出数据集包含的所有类别标签数
print len(data_set.label)
# 输出数据集包含的文件内容
print len(data_set.contents)
