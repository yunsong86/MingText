#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: Feature.py
@time: 9/27/17 7:50 PM
@desc: python3.6
"""
import os
from Log import *
import time
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from chardet.universaldetector import UniversalDetector

def detectCode(filename):
    detector = UniversalDetector()  # 初始化一个UniversalDetector对象
    f = open(filename, 'rb')  # test.txt是一个utf-8编码的文本文档
    for line in f:
        detector.feed(line)  # 逐行载入UniversalDetector对象中进行识别
        if detector.done:  # done为一个布尔值，默认为False，达到阈值时变为True
            break

    detector.close()  # 调用该函数做最后的数据整合
    f.close()
    codetype = detector.result['encoding']
    if codetype:
        if codetype.startswith('utf'):
            return 'utf'
        else:
            return 'gbk'
    else:

        return  'gbk'




def onefileGBK2UTF(filename):
    typeCode = detectCode(filename)
    print('>>>>>>>>>>>>>>>'+filename+'<<<<<<<<<<<<<<<<<<<<<<')
    print('****'+typeCode+'****')
    if typeCode == 'gbk':
        try:
            with open(filename, 'rb')as file_obj:
                content = file_obj.read().decode(encoding='gbk',errors='ignore')
            with open(filename,'w', encoding='utf-8') as file_obj:
                file_obj.write(content)

        except Exception as e:
            print('************' + filename + '************')
            print(e)
        finally:
           pass

    else:
        try:
            with open(filename, 'rb') as file_obj:
                content = file_obj.read().decode(encoding='utf-8',errors='ignore')
            with open(filename,'w', encoding='utf-8') as file_obj:
                file_obj.write(content)
        except Exception as e:
            print('************' + filename + '************')
            print(e)
        finally:
            # os.remove(filename)
            pass






def filesGBK2UTF(corpusdir):
    for root, dirs, files in os.walk(corpusdir):
        for name in files:
            filename = os.path.join(root, name)
            onefileGBK2UTF(filename)



def corpus(corpusdir):
    # data = load_files(corpusdir,encoding='utf-8',decode_error='ignore')
    data = load_files(corpusdir,encoding='utf-8')

    return data



def stopwords(filename):
    with open(filename, 'r',encoding='utf-8') as f:
        return [w.strip() for w in f]

def BOW(documents, vocabulary=None):
    vectorizer = CountVectorizer(vocabulary=vocabulary,stop_words=stopwords())
    if vocabulary:
        bow = vectorizer.transform(documents)
    else:
        bow = vectorizer.fit_transform(documents)
    return vectorizer.get_feature_names(), bow


def Tfidf(bow):
    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(bow)
    return tf_idf

if __name__ == "__main__":
    corpusdir = '/mnt/hgfs/UbunutWin/corpus/text_classification_fudan_test'
    # filesGBK2UTF(corpusdir=corpusdir)
    # data = corpus(corpusdir=corpusdir)
    logger.info('test')
    # filename = '/mnt/hgfs/UbunutWin/2.txt'
    # onefileGBK2UTF(filename)
    # filesGBK2UTF(corpusdir=corpusdir)
    print('dd')