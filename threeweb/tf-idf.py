#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@license: (C) Copyright 2013-2017,Bertadata Corporation Limited.
@contact: yunsong@bertadata.com
@file: tf-idf.py
@time: 2017/9/10 17:09
@desc: python2
"""
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import sys
import os
import datetime
# 引入Bunch类
from sklearn.datasets.base import Bunch
# 引入持久化类
import pickle
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

# 导入训练语料
data_set = {}  # 清空原有数据集
# 训练语料集路径
train_path = "E:/NLP_DATA_BERTA/corpus/threeweb/text_mining/text_corpus_wordbag/train_set.data"
file_obj = open(train_path, "rb")
# 读取持久化后的对象
data_set = pickle.load(file_obj)
file_obj.close()
# 定义词袋数据结构
# tdm:tf-idf计算后词袋
wordbag = Bunch(target_name=[], label=[], filenames=[], tdm=[], vocabulary={})
wordbag.target_name = data_set.target_name
wordbag.label = data_set.label
wordbag.filenames = data_set.filenames
# 构建语料
corpus = data_set.contents

# 从文件导入停用词表
stpwrdpath = "E:/NLP_DATA_BERTA/resouces/stopwords.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
# 将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

# 计算词袋创建时间：获取开始时间
start = datetime.datetime.now()
# 使用 TfidfVectorizer初始化向量空间模型--创建词袋
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=stpwrdlst)
# 该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()
# 文本转为词频矩阵
fea_train = vectorizer.fit_transform(corpus)

# 计算词袋时间：结束时间
end = datetime.datetime.now()
print 'create word bag peroid:', (end - start).seconds, "seconds"

# 计算词袋的行列数
print 'Size of fea_train:\n', fea_train.shape  # fea_train[0:10,0:10]
# 为tdm赋值
wordbag.tdm = fea_train
wordbag.vocabulary = vectorizer.vocabulary_
# 创建词袋的持久化
file_obj = open("E:/NLP_DATA_BERTA/corpus/threeweb/text_mining/text_corpus_wordbag/word_bag.data", "wb")
pickle.dump(wordbag, file_obj)
file_obj.close()
