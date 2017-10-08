#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: classifyNewsBaseOnSVM.py
@time: 2017/10/7 21:45
@desc: python3.6
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.model_selection import *

import logging
from logging.handlers import RotatingFileHandler

logging.root.setLevel(level=logging.INFO)
Rthandler = RotatingFileHandler(filename='classifyNewsBaseOnSVM.log', mode='a', maxBytes=1 * 1024 * 1024, backupCount=2)
Rthandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
Rthandler.setFormatter(formatter)
logging.getLogger('').addHandler(Rthandler)

##################################################
#定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


logger = logging.getLogger()




def load_new_data(corpus_dir):
    news = load_files(corpus_dir, encoding='utf-8')
    tfidf_vect = TfidfVectorizer()
    X = tfidf_vect.fit_transform(news.data)
    X_train, X_test, y_train, y_test = train_test_split(X, news.target, test_size=0.3, stratify=news.target)
    return X_train, X_test, y_train, y_test


def grid_SVC():
    svc_tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000], },
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                            {'kernel': ['poly']},
                            {'kernel': ['sigmoid']}]


def classify_SVC(*data):
    X_train, X_test, y_train, y_test = data
    kernels = ['rbf', 'linear', 'sigmoid', 'poly']
    for kernel in kernels:
        logger.info('kernel:' + kernel)
        cls = SVC(kernel=kernel, class_weight='balanced')
        cls.fit(X_train, y_train)
        y_predict = cls.predict(X_test)
        logger.info(classification_report(y_test, y_predict))
        logger.info('Score: %.2f' % cls.score(X_test, y_test))


def grid_LinearSVC():
    scores = ['precision', 'recall']
    parameters = [{'C': [1, 10, 100, 1000],
                              'multi_class': ['ovr', 'crammer_singer'],
                              'loss': ['squared_hinge', 'squared_hinge']}]
    for score in scores:
        clf = GridSearchCV(LinearSVC(), parameters, cv=4, scoring='%s_weighted' % score)
        clf.fit(X_train, y_train)
        logger.info("Best parameters set found on development set:")
        logger.info(clf.best_params_)
        logger.info("Detailed classification report:")
        logger.info("The scores are computed on the full evaluation set.")

        y_true, y_pred = y_test, clf.predict(X_test)
        logger.info(classification_report(y_true, y_pred))


def classify_LinearSVC(*data):
    logger.info('*****LinearSVC*****')
    X_train, X_test, y_train, y_test = data
    cls = LinearSVC(class_weight='balanced')
    cls.fit(X_train, y_train)
    y_predict = cls.predict(X_test)
    logger.info(classification_report(y_test, y_predict))
    logger.info('Score: %.2f' % cls.score(X_test, y_test))


if __name__ == "__main__":
    corpus_dir = 'D:/UbunutWin/corpus/news_data/BQ20_seg'
    X_train, X_test, y_train, y_test = load_new_data(corpus_dir)
    # classify_LinearSVC(X_train, X_test, y_train, y_test)
    classify_SVC(X_train, X_test, y_train, y_test)
