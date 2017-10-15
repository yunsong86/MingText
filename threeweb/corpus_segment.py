#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@license: (C) Copyright 2013-2017,Bertadata Corporation Limited.
@contact: yunsong@bertadata.com
@file: corpus_segment.py
@time: 2017/9/10 17:02
@desc: python2
"""
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


import sys
import os
import jieba

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')
# 分类语料库路径
corpus_path = "E:/NLP_DATA_BERTA/corpus/threeweb/text_mining/text_corpus_small"+"/"
# 分词后分类语料库路径
seg_path = "E:/NLP_DATA_BERTA/corpus/threeweb/text_mining/text_corpus_segment"+"/"

# 获取corpus_path下的所有子目录
dir_list = os.listdir(corpus_path)

# 获取每个目录下所有的文件
for mydir in dir_list:
        class_path = corpus_path+mydir+"/" # 拼出分类子目录的路径
        file_list = os.listdir(class_path)  # 获取class_path下的所有文件
        for file_path in file_list:   # 遍历所有文件
                file_name = class_path + file_path  # 拼出文件名全路径
                file_read = open(file_name, 'rb')   # 打开一个文件
                raw_corpus = file_read.read()       # 读取未分词语料
                seg_corpus = jieba.cut(raw_corpus)  # 结巴分词操作
                #拼出分词后语料分类目录
                seg_dir = seg_path+mydir+"/"
                if not os.path.exists(seg_dir):    #如果没有创建
                        os.makedirs(seg_dir)
                file_write = open ( seg_dir + file_path, 'wb' ) #创建分词后语料文件，文件名与未分词语料相同
                file_write.write(" ".join(seg_corpus))  #用空格将分词结果分开并写入到分词后语料文件中
                file_read.close()  #关闭打开的文件
                file_write.close()  #关闭写入的文件

print "中文语料分词成功完成！！！"



