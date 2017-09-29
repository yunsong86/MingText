#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: try.py
@time: 17-9-28 下午8:41
@desc: python3.6
"""
from TextPreprocess import TextPreprocess
tp = TextPreprocess()
tp.corpus_path = "/mnt/hgfs/UbunutWin/corpus/text_fudan/classification_fudan_corpus"

tp.pos_path = "/mnt/hgfs/UbunutWin/corpus/text_fudan/classification_fudan_pos"       #预处理后语料路径
tp.segment_path = '/mnt/hgfs/UbunutWin/corpus/text_fudan/classification_fudan_seg'   #分词后语料路径
tp.model_path = "/mnt/hgfs/UbunutWin/corpus/text_fudan/model"   #词袋模型路径
tp.stopword_path =  "/mnt/hgfs/UbunutWin/resources/stopwords.txt"  #停止词路径
tp.trainset_file_name = "trainset.dat"      #训练集文件名
tp.tfidf_wordbag_file_name = "tfidfwordbag.dat"       #词包文件名
tp.preprocess()
tp.segment()
tp.train_bag()
tp.tfidf_bag()
tp.verify_trainset()
tp.verify_wordbag()