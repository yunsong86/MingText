#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@license: (C) Copyright 2013-2017,Bertadata Corporation Limited.
@contact: yunsong@bertadata.com
@file: text_mining.py
@time: 2017/9/10 17:19
@desc: python2
"""
import sys

reload(sys)
sys.setdefaultencoding('utf-8')



import sys
import os
import warnings
import numpy as np
from sklearn import metrics

warnings.filterwarnings("ignore")

# 精度测试
def calculate_accurate(actual,predict):
        m_precision = metrics.accuracy_score(actual,predict)
        print '结果计算:'
        print '精度:{0:.3f}'.format(m_precision)

# 召回，精度，f1测试
def calculate_result(actual,predict):
    pass
        # m_precision = metrics.precision_score(actual,predict)
        # m_recall = metrics.recall_score(actual,predict)
        # print '结果计算:'
        # print '精度:{0:.3f}'.format(m_precision)
        # print '召回:{0:0.3f}'.format(m_recall)
        # print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual,predict))

# 综合测试报告
def test_report(actual,predicted,category):
        print(metrics.classification_report(actual, predicted,target_names=category))