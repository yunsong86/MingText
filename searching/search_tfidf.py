#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: search_tfidf.py
@time: 2017/10/15 9:37
@desc: python3.6
"""


import logging
from sklearn.datasets import load_files
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from logging.handlers import RotatingFileHandler

logging.root.setLevel(level=logging.INFO)
Rthandler = RotatingFileHandler(filename='classifyNewsBaseOnGDBT.log', mode='a', maxBytes=1 * 1024 * 1024,
                                backupCount=2)
Rthandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
Rthandler.setFormatter(formatter)
logging.getLogger('').addHandler(Rthandler)

##################################################
# 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logger = logging.getLogger()

def min_df_tfidf(corpus, mindf=None):
    if not corpus:
        logging.info("No corpus")
    news = load_files(corpus, encoding='utf-8')
    vector = TfidfVectorizer(min_df=mindf)
    tfidf_vect = vector.fit_transform(news.data)
    feature_names = vector.get_feature_names()
    vocabulary = vector.vocabulary_






def max_df_tfidf(corpus, maxdf):
    pass

def max_features_tfidf(coupus, maxFeatures):
    pass


if __name__ == '__main__':
    corpus = 'D:/UbuntuWin/corpus/text_fudan/classification_fudan_seg'
    min_df_tfidf(corpus)