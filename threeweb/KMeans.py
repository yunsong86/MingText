#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@license: (C) Copyright 2013-2017,Bertadata Corporation Limited.
@contact: yunsong@bertadata.com
@file: KMeans.py
@time: 2017/9/10 17:35
@desc: python2
"""
import sys

reload(sys)
sys.setdefaultencoding('utf-8')



import sys
import os
import numpy as np
#引入Bunch类
from sklearn.datasets.base import Bunch
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from TextPreprocess import TextPreprocess  # 第一个是文件名，第二个是类名
#导入KMeans算法
from sklearn.cluster import KMeans

from text_mining import calculate_result,calculate_accurate


# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

# 导入语料集
corpus_set = TextPreprocess()
corpus_set.wordbag_path = "text_corpus1_wordbag/"   #词袋模型路径
corpus_set.trainset_name = "train_set.data"       #词包文件名
corpus_set.stopword_path = "extra_dict/hlt_stop_words.txt"

#从文件导入停用词表
stpwrdlst = corpus_set.getStopword(corpus_set.stopword_path)

#从文件导入数据包
corpus_set.load_trainset()
clusters = len(corpus_set.data_set.target_name)
print "共",clusters,"种类别:",corpus_set.data_set.target_name

# 计算 tf-idf权值
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words=stpwrdlst,use_idf=True,max_features=10000)
feat_test = vectorizer.fit_transform(corpus_set.data_set.contents)

#应用Kmeans算法 输入词袋向量和类别标签
predict = KMeans(n_clusters = clusters,init='k-means++', max_iter=100, n_init=1)
# 估计聚类结果
predict.fit(feat_test)


# 输出聚类结果
calculate_result(corpus_set.data_set.label,predict.labels_);