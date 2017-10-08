#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: classifyNewsBaseOnGDBT.py
@time: 2017/10/7 21:39
@desc: python3.6
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import ensemble
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from Log import *


def load_new_data(corpus_dir):
    news = load_files(corpus_dir, encoding='utf-8')
    tfidf_vect = TfidfVectorizer(max_features=100000)
    X = tfidf_vect.fit_transform(news.data)

    # # Don't need both X and transformer; they should be identical
    # matrix_terms = np.array(tfidf_vect.get_feature_names())
    #
    # # Use the axis keyword to sum over rows
    # matrix_freq = np.asarray(X.sum(axis=0)).ravel()
    # final_matrix = np.array([matrix_terms, matrix_freq])

    X_train, X_test, y_train, y_test = train_test_split(X, news.target, test_size=0.3, stratify=news.target)
    return X_train, X_test, y_train, y_test


def do_GradientBoostingClassifier(*data):
    '''
    测试 GradientBoostingClassifier 的用法

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train, X_test, y_train, y_test = data
    clf = ensemble.GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    print("Traing Score:%f" % clf.score(X_train, y_train))
    print("Testing Score:%f" % clf.score(X_test, y_test))


def do_GradientBoostingClassifier_num(*data):
    '''
    测试 GradientBoostingClassifier 的预测性能随 n_estimators 参数的影响

    :param data:   可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train, X_test, y_train, y_test = data
    nums = np.arange(1, 100, step=2)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    testing_scores = []
    training_scores = []
    for num in nums:
        clf = ensemble.GradientBoostingClassifier(n_estimators=num)
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train, y_train))
        testing_scores.append(clf.score(X_test, y_test))
    ax.plot(nums, training_scores, label="Training Score")
    ax.plot(nums, testing_scores, label="Testing Score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    plt.suptitle("GradientBoostingClassifier")
    plt.show()


def do_GradientBoostingClassifier_maxdepth(*data):
    '''
    测试 GradientBoostingClassifier 的预测性能随 max_depth 参数的影响

    :param data:    可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train, X_test, y_train, y_test = data
    maxdepths = np.arange(1, 20)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    testing_scores = []
    training_scores = []
    for maxdepth in maxdepths:
        clf = ensemble.GradientBoostingClassifier(max_depth=maxdepth, max_leaf_nodes=None)
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train, y_train))
        testing_scores.append(clf.score(X_test, y_test))
    ax.plot(maxdepths, training_scores, label="Training Score")
    ax.plot(maxdepths, testing_scores, label="Testing Score")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    plt.suptitle("GradientBoostingClassifier")
    plt.show()


def do_GradientBoostingClassifier_learning(*data):
    '''
    测试 GradientBoostingClassifier 的预测性能随学习率参数的影响

    :param data:    可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train, X_test, y_train, y_test = data
    learnings = np.linspace(0.01, 1.0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    testing_scores = []
    training_scores = []
    for learning in learnings:
        clf = ensemble.GradientBoostingClassifier(learning_rate=learning)
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train, y_train))
        testing_scores.append(clf.score(X_test, y_test))
    ax.plot(learnings, training_scores, label="Training Score")
    ax.plot(learnings, testing_scores, label="Testing Score")
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    plt.suptitle("GradientBoostingClassifier")
    plt.show()


def do_GradientBoostingClassifier_subsample(*data):
    '''
    测试 GradientBoostingClassifier 的预测性能随 subsample 参数的影响

    :param data:    可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train, X_test, y_train, y_test = data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    subsamples = np.linspace(0.01, 1.0)
    testing_scores = []
    training_scores = []
    for subsample in subsamples:
        clf = ensemble.GradientBoostingClassifier(subsample=subsample)
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train, y_train))
        testing_scores.append(clf.score(X_test, y_test))
    ax.plot(subsamples, training_scores, label="Training Score")
    ax.plot(subsamples, testing_scores, label="Training Score")
    ax.set_xlabel("subsample")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    plt.suptitle("GradientBoostingClassifier")
    plt.show()


def do_GradientBoostingClassifier_max_features(*data):
    '''
    测试 GradientBoostingClassifier 的预测性能随 max_features 参数的影响

    :param data:    可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:   None
    '''
    X_train, X_test, y_train, y_test = data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    max_features = np.linspace(0.01, 1.0)
    testing_scores = []
    training_scores = []
    for features in max_features:
        clf = ensemble.GradientBoostingClassifier(max_features=features)
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train, y_train))
        testing_scores.append(clf.score(X_test, y_test))
    ax.plot(max_features, training_scores, label="Training Score")
    ax.plot(max_features, testing_scores, label="Training Score")
    ax.set_xlabel("max_features")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    plt.suptitle("GradientBoostingClassifier")
    plt.show()


if __name__ == '__main__':
    corpus_dir = 'D:/UbunutWin/corpus/news_data/BQ20_seg'
    logger.info("******GDBT CALSSIFY BQ20 NEWS******")
    news = load_files(corpus_dir, encoding='utf-8')
    for n in range(1900, 200000, 1000):
        s = datetime.now()
        tfidf_vect = TfidfVectorizer(max_features=n)
        X = tfidf_vect.fit_transform(news.data)
        logger.info("-----n_features-----"+str(n))
        logger.info(X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, news.target, test_size=0.3, stratify=news.target)
        X_train = X_train.toarray()
        X_test = X_test.toarray()
        logger.info("complete csr_matrix toarray")
        do_GradientBoostingClassifier(X_train, X_test, y_train, y_test)  # 调用 do_GradientBoostingClassifier
        e = datetime.now()
        logger.info("耗时： "+str((e-s).seconds))
        # do_GradientBoostingClassifier_num(X_train,X_test,y_train,y_test) # 调用 do_GradientBoostingClassifier_num
        # do_GradientBoostingClassifier_maxdepth(X_train,X_test,y_train,y_test) # 调用 do_GradientBoostingClassifier_maxdepth
        # do_GradientBoostingClassifier_learning(X_train,X_test,y_train,y_test) # 调用 do_GradientBoostingClassifier_learning
        # do_GradientBoostingClassifier_subsample(X_train,X_test,y_train,y_test) # 调用 do_GradientBoostingClassifier_subsample
        # do_GradientBoostingClassifier_max_features(X_train,X_test,y_train,y_test) # 调用 do_GradientBoostingClassifier_max_features
