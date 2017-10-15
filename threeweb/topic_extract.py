#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@license: (C) Copyright 2013-2017,Bertadata Corporation Limited.
@contact: yunsong@bertadata.com
@file: topic_extract.py
@time: 2017/9/10 17:40
@desc: python2
"""
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import sys
import os
import warnings
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups
from TextPreprocess import TextPreprocess

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')
# 忽略警告
warnings.filterwarnings("ignore")

n_samples = 2000
n_features = 1000
n_top_words = 20

# 导入语料集
corpus_set = TextPreprocess()
corpus_set.wordbag_path = "text_corpus1_wordbag/"  # 词袋模型路径
corpus_set.trainset_name = "train_set.data"  # 词包文件名
corpus_set.stopword_path = "extra_dict/hlt_stop_words.txt"

# 从文件导入停用词表
stpwrdlst = corpus_set.getStopword(corpus_set.stopword_path)

# 从文件导入数据包
corpus_set.load_trainset()
clusters = len(corpus_set.data_set.target_name)

print "共", clusters, "种类别:", corpus_set.data_set.target_name

for i in range(0, clusters - 1):
    findx = corpus_set.data_set.label.index(i)
    counts = corpus_set.data_set.label.count(i)
    lindx = findx + counts - 1

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                 stop_words=stpwrdlst)
    tfidf = vectorizer.fit_transform(corpus_set.data_set.contents[findx:lindx])
    # Fit the NMF model
    nmf = NMF(n_components=1, random_state=1).fit(tfidf)

    feature_names = vectorizer.get_feature_names()

    # print "nmf.components_:",len(nmf.components_)
    print("Topic :", corpus_set.data_set.target_name[i])
    print(" ".join([feature_names[i]
                    for i in nmf.components_[0].argsort()[:-n_top_words - 1:-1]]))
