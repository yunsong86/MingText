#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: Feature.py
@time: 9/27/17 7:50 PM
@desc: python3.6
"""
import os
from com_resources import *
from Log import *
from sklearn.datasets.base import Bunch
import time
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from chardet.universaldetector import UniversalDetector


def detectCode(filename):
    detector = UniversalDetector()  # 初始化一个UniversalDetector对象
    f = open(filename, 'rb')
    for line in f:
        detector.feed(line)  # 逐行载入UniversalDetector对象中进行识别
        if detector.done:  # done为一个布尔值，默认为False，达到阈值时变为True
            break

    detector.close()  # 调用该函数做最后的数据整合
    f.close()
    codetype = detector.result['encoding']
    if codetype:
        print('****' + codetype + '****')
        if codetype.startswith('utf'):
            logger.info(filename + ' : codetype UTF-8')
            return 'utf'
        else:
            return 'gbk'
    else:
        logger.info(filename + ' : codetype None')
        return 'gbk'


def onefileGBK2UTF(filename):
    typeCode = detectCode(filename)
    print('>>>>>>>>>>>>>>>' + filename + '<<<<<<<<<<<<<<<<<<<<<<')
    if typeCode == 'gbk':
        try:
            with open(filename, 'rb')as file_obj:
                content = file_obj.read().decode(encoding='gbk', errors='ignore')
            with open(filename, 'w', encoding='utf-8') as file_obj:
                file_obj.write(content)

        except Exception as e:
            print('************' + filename + '************')
            print(e)
            logger.error(filename + ": " + e)
        finally:
            pass

    else:
        try:
            with open(filename, 'rb') as file_obj:
                content = file_obj.read().decode(encoding='utf-8', errors='ignore')
            with open(filename, 'w', encoding='utf-8') as file_obj:
                file_obj.write(content)
        except Exception as e:
            print('************' + filename + '************')
            print(e)
            logger.error(filename + ": " + e)

        finally:
            # os.remove(filename)
            pass


def filesGBK2UTF(corpusdir):
    for root, dirs, files in os.walk(corpusdir):
        for name in files:
            filename = os.path.join(root, name)
            onefileGBK2UTF(filename)

def loadStopWords(filename):
    file_obj = open(filename, 'r', encoding='utf-8')
    stopword_list = [w.strip() for w in file_obj if w]
    file_obj.close()
    return stopword_list


def segText(filename):
    if not STOP_WORDS_LIST:
        STOP_WORDS_LIST = loadStopWords(STOP_WORDS_DIR)
    with open(filename, 'r', encoding='utf-8') as file_obj:
        return [w.strip() for w in file_obj if w not in STOP_WORDS_LIST and w]


def corpus(corpusdir):
    corpus_Bunch = Bunch(targetName=[], label=[], segContent=[])

    dir_list = os.listdir(corpusdir)
    corpus_Bunch.targetName = dir_list
    for mydir in dir_list:
        class_path = corpusdir + '/' + mydir + '/'
        filename_list = os.listdir(class_path)
        for name in filename_list:
            file_name = class_path + name
            corpus_Bunch.label.append(corpus_Bunch.target_name.index(mydir))
            corpus_Bunch.segContent.append(segText(file_name))
            print(file_name)


            # data = load_files(corpusdir, encoding='utf-8', decode_error='ignore')
            # data = load_files(corpusdir, encoding='utf-8')
            # return data





def BOW(documents, vocabulary=None):
    vectorizer = CountVectorizer(vocabulary=vocabulary, stop_words=stopwords())
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
    data = corpus(corpusdir=corpusdir)
    # filename = '/mnt/hgfs/UbunutWin/2.txt'
    # onefileGBK2UTF(filename)
    # filesGBK2UTF(corpusdir=corpusdir)
    print('done')
