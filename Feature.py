#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: Feature.py
@time: 9/27/17 7:50 PM
@desc: python3.6
"""
import os
import jieba
import pickle
from com_resources import *
from Log import *
from sklearn.datasets.base import Bunch
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
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
    global STOP_WORDS_LIST
    if STOP_WORDS_LIST == None:
        STOP_WORDS_LIST = loadStopWords(STOP_WORDS_DIR)
    with open(filename, 'r', encoding='utf-8') as file_obj:
        content = file_obj.read()
    return [w.strip() for w in jieba.cut(content) if w not in STOP_WORDS_LIST and len(w.strip()) > 0]


def corpus(corpusdir):
    starttime = datetime.now()
    corpus_Bunch = Bunch(targetName=[], label=[], content=[])
    dir_list = os.listdir(corpusdir)
    corpus_Bunch.targetName = dir_list
    for mydir in dir_list:
        class_path = corpusdir + '/' + mydir + '/'
        filename_list = os.listdir(class_path)
        for name in filename_list:
            file_name = class_path + name
            corpus_Bunch.label.append(corpus_Bunch.targetName.index(mydir))
            corpus_Bunch.content.append(' '.join(segText(file_name)))
            print(file_name)

    endtime = datetime.now()
    logger.info('分词完成'+str((starttime-endtime).seconds))
    return corpus_Bunch


def tfidf_bag(corpus_Bunch,filename):

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus_Bunch.content)
    vocabulary = vectorizer.vocabulary_

    tfidf_Bunch = Bunch(targetName=[], label=[], tfidf=[], vocabulary={})
    tfidf_Bunch.targetName = corpus_Bunch.targetName
    tfidf_Bunch.label = corpus_Bunch.label
    tfidf_Bunch.tfidf = tfidf
    tfidf_Bunch.vocabulary = vocabulary

    file_obj = open(filename, "wb")
    pickle.dump(tfidf_Bunch, file_obj)
    file_obj.close()


def tfidf_value(test_data, myvocabulary):
    vectorizer = TfidfVectorizer(vocabulary=myvocabulary)
    return vectorizer.fit_transform(test_data)


def savePick(filename, dataBunch):
    file_obj = open(filename, "wb")
    pickle.dump(dataBunch, file_obj)
    file_obj.close()


def loadPick(filename):
    file_obj = open(filename, 'rb')
    pk = pickle.load(file_obj)
    file_obj.close()
    return pk


def train_tfidf(corpus, stpwrdlst):
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=stpwrdlst)
    fea_train = vectorizer.fit_transform(corpus)
    vocabulary = vectorizer.vocabulary_
    return fea_train, vocabulary


def train_tfidf_Bunch(corpus_Bunch, stpwrdlst):
    fea_train, vocabulary = train_tfidf(corpus, stpwrdlst)


# def BOW(documents, vocabulary=None):
#     global STOP_WORDS_LIST
#     if STOP_WORDS_LIST == None:
#         STOP_WORDS_LIST = loadStopWords(STOP_WORDS_DIR)
#
#     vectorizer = CountVectorizer(vocabulary=vocabulary, stop_words=STOP_WORDS_LIST)
#     if vocabulary:
#         bow = vectorizer.transform(documents)
#     else:
#         bow = vectorizer.fit_transform(documents)
#     return vectorizer.get_feature_names(), bow
#
#
# def Tfidf(bow, vocabulary=None):
#     transformer = TfidfTransformer()
#     tf_idf = transformer.fit_transform(bow)
#     return tf_idf


if __name__ == "__main__":
    corpusdir = '/mnt/hgfs/UbunutWin/corpus/text_classification_fudan_test'
    # filesGBK2UTF(corpusdir=corpusdir)
    # seg_corpus_Bunch = corpus(corpusdir=corpusdir)
    dest = '/mnt/hgfs/UbunutWin/data/text_classification_fudan_test.seg.pkl'
    # savePick(dest, seg_corpus_Bunch)
    seg_corpus_Bunch = loadPick(dest)
    vectorizer = TfidfVectorizer()
    fea_train = vectorizer.fit_transform(seg_corpus_Bunch.content)


    # filename = '/mnt/hgfs/UbunutWin/2.txt'
    # onefileGBK2UTF(filename)
    # filesGBK2UTF(corpusdir=corpusdir)
    print('done')
