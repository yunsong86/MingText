#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: donewclassify.py
@time: 17-9-30 上午11:30
@desc: python3.6
"""

from DoNewClassifyTextPreprocess import DoNewClassifyTextPreprocess
tp = DoNewClassifyTextPreprocess()

tp.corpus_path = "/mnt/hgfs/UbunutWin/corpus/news_data"
tp.pos_path = "/mnt/hgfs/UbunutWin/corpus/news_data"       #预处理后语料路径
tp.segment_path = '/mnt/hgfs/UbunutWin/corpus/news_data_seg'   #分词后语料路径
tp.model_path = "/mnt/hgfs/UbunutWin/corpus/news_data_model"   #词袋模型路径
tp.stopword_path =  "/mnt/hgfs/UbunutWin/resources/stopwords.txt"  #停止词路径
tp.trainset_file_name = "trainset.dat"      #训练集文件名
tp.tfidf_wordbag_file_name = "tfidfwordbag.dat"       #词包文件名

# tp.preprocess()
# tp.segment()
tp.train_bag()
tp.tfidf_bag()
tp.verify_trainset()
tp.verify_wordbag()