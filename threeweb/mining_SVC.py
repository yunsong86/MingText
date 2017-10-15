#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@license: (C) Copyright 2013-2017,Bertadata Corporation Limited.
@contact: yunsong@bertadata.com
@file: mining_SVC.py
@time: 2017/9/10 17:27
@desc: python2
"""
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
# 引入Bunch类
from sklearn.datasets.base import Bunch
# 引入持久化类
import pickle
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from TextPreprocess import TextPreprocess  # 第一个是文件名，第二个是类名
# 导入线性核svm算法
from sklearn.svm import LinearSVC

from text_mining import calculate_result, calculate_accurate, test_report

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

# 测试语料预处理
testsamp = TextPreprocess()
# testsamp.corpus_path = "test_corpus1_small/"    #原始语料路径
# testsamp.pos_path = "test_corpus1_pos/"       #预处理后语料路径
# 测试语料预处理
# testsamp.preprocess()


testsamp.segment_path = "E:/NLP_DATA_BERTA/corpus/threeweb/text_mining/test_corpus_segment/"  # 分词后语料路径
testsamp.stopword_path = "E:/NLP_DATA_BERTA/resouces/stopwords.txt" # 停止词路径
# 为测试语料分词
# testsamp.segment()

# 实际应用中可直接导入分词后测试语料
testsamp.wordbag_path = "E:/NLP_DATA_BERTA/corpus/threeweb/text_mining/text_corpus_wordbag/"  # 词袋模型路径
testsamp.trainset_name = "train_set.data"  # 训练集文件名
# testsamp.train_bag()

# 加载全部测试语料
testsamp.load_trainset()

# 对测试文本进行tf-idf计算
# 从文件导入停用词表
stpwrdlst = testsamp.getStopword(testsamp.stopword_path)
print len(testsamp.data_set.contents)

# 导入训练词袋模型
train_set = TextPreprocess()
train_set.wordbag_path = "E:/NLP_DATA_BERTA/corpus/threeweb/text_mining/text_corpus_wordbag/"
train_set.wordbag_name = "word_bag.data"  # 词袋文件名
train_set.load_wordbag()
print train_set.wordbag.tdm.shape

# 计算测试集的tfidf_value特征
fea_test = testsamp.tfidf_value(testsamp.data_set.contents, stpwrdlst, train_set.wordbag.vocabulary)
print fea_test.shape

# 应用linear_svm算法 输入词袋向量和分类标签
# svclf = SVC(kernel = 'linear')   # default with 'rbf'
svclf = LinearSVC(penalty="l1", dual=False, tol=1e-4)
# 训练分类器
svclf.fit(train_set.wordbag.tdm, train_set.wordbag.label)
# 预测分类结果
predicted = svclf.predict(fea_test)

# 测试集与训练集详细比较
# i=0
# for file_name,expct_cate in zip(testsamp.data_set.label,predicted):
#        print "测试语料文件名:",testsamp.data_set.filenames[i],": 实际类别:",testsamp.data_set.target_name[testsamp.data_set.label[i]],"<-->预测类别:",train_set.wordbag.target_name[expct_cate]
#        i +=1

# list转np.array
actual = np.array(testsamp.data_set.label)

# 计算分类各种参数指标
calculate_result(actual, predicted)

# 综合测试报告
test_report(actual, predicted, testsamp.data_set.target_name)
