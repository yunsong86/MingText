#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@license: (C) Copyright 2013-2017,Bertadata Corporation Limited.
@contact: yunsong@bertadata.com
@file: textmining.py
@time: 2017/9/10 17:14
@desc: python2
"""
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import sys
import os
from TextPreprocess import TextPreprocess  # 第一个是文件名，第二个是类名

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')
# 实例化这个类
tp = TextPreprocess()
tp.corpus_path = "E:/NLP_DATA_BERTA/corpus/threeweb/text_mining/text_corpus_small/"    #原始语料路径
tp.pos_path = "text_corpus_pos/"       #预处理后语料路径
tp.segment_path = "E:/NLP_DATA_BERTA/corpus/threeweb/text_mining/text_corpus_segment/"   #分词后语料路径
tp.wordbag_path = "E:/NLP_DATA_BERTA/corpus/threeweb/text_mining/text_corpus_wordbag/"   #词袋模型路径
tp.stopword_path =  "E:/NLP_DATA_BERTA/resouces/stopwords.txt"  #停止词路径
tp.trainset_name = "trainset.dat"      #训练集文件名
tp.wordbag_name = "wordbag.dat"       #词包文件名
tp.preprocess()
tp.segment()
tp.train_bag()
tp.tfidf_bag()
tp.verify_trainset()
tp.verify_wordbag()

