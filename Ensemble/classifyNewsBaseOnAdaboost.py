#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: classifyNewsBaseOnAdaboost.py
@time: 2017/10/7 21:39
@desc: python3.6
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import ensemble
import matplotlib.pyplot as  plt
import numpy as np


def load_new_data(corpus_dir):
    news = load_files(corpus_dir, encoding='utf-8')
    tfidf_vect = TfidfVectorizer()
    X = tfidf_vect.fit_transform(news.data)
    X_train, X_test, y_train, y_test = train_test_split(X, news.target, test_size=0.3, stratify=news.target)
    return X_train, X_test, y_train, y_test


def do_AdaBoostClassifier(*data):
    '''
    测试 AdaBoostClassifier 的用法，绘制 AdaBoostClassifier 的预测性能随基础分类器数量的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train, X_test, y_train, y_test = data
    clf = ensemble.AdaBoostClassifier(learning_rate=0.1)
    clf.fit(X_train, y_train)
    ## 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    estimators_num = len(clf.estimators_)
    X = range(1, estimators_num + 1)
    ax.plot(list(X), list(clf.staged_score(X_train, y_train)), label="Traing score")
    ax.plot(list(X), list(clf.staged_score(X_test, y_test)), label="Testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="best")
    ax.set_title("AdaBoostClassifier")
    plt.show()

# GaussianNB 不能用在文本分类上
def do_AdaBoostClassifier_base_classifier(*data):
    '''
    测试  AdaBoostClassifier 的预测性能随基础分类器数量和基础分类器的类型的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    from sklearn.naive_bayes import GaussianNB
    X_train, X_test, y_train, y_test = data
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ########### 默认的个体分类器 #############
    clf = ensemble.AdaBoostClassifier(learning_rate=0.1)
    clf.fit(X_train, y_train)
    ## 绘图
    estimators_num = len(clf.estimators_)
    X = range(1, estimators_num + 1)
    ax.plot(list(X), list(clf.staged_score(X_train, y_train)), label="Traing score")
    ax.plot(list(X), list(clf.staged_score(X_test, y_test)), label="Testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    ax.set_title("AdaBoostClassifier with Decision Tree")
    ####### Gaussian Naive Bayes 个体分类器 ########
    ax = fig.add_subplot(2, 1, 2)
    clf = ensemble.AdaBoostClassifier(learning_rate=0.1, base_estimator=GaussianNB())
    clf.fit(X_train, y_train)
    ## 绘图
    estimators_num = len(clf.estimators_)
    X = range(1, estimators_num + 1)
    ax.plot(list(X), list(clf.staged_score(X_train, y_train)), label="Traing score")
    ax.plot(list(X), list(clf.staged_score(X_test, y_test)), label="Testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    ax.set_title("AdaBoostClassifier with Gaussian Naive Bayes")
    plt.show()


def do_AdaBoostClassifier_learning_rate(*data):
    '''
    测试  AdaBoostClassifier 的预测性能随学习率的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train, X_test, y_train, y_test = data
    learning_rates = np.linspace(0.01, 1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    traing_scores = []
    testing_scores = []
    for learning_rate in learning_rates:
        clf = ensemble.AdaBoostClassifier(learning_rate=learning_rate, n_estimators=500)
        clf.fit(X_train, y_train)
        traing_scores.append(clf.score(X_train, y_train))
        testing_scores.append(clf.score(X_test, y_test))
    ax.plot(learning_rates, traing_scores, label="Traing score")
    ax.plot(learning_rates, testing_scores, label="Testing score")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("score")
    ax.legend(loc="best")
    ax.set_title("AdaBoostClassifier")
    plt.show()


def do_AdaBoostClassifier_algorithm(*data):
    '''
    测试  AdaBoostClassifier 的预测性能随学习率和 algorithm 参数的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train, X_test, y_train, y_test = data
    algorithms = ['SAMME.R', 'SAMME']
    fig = plt.figure()
    learning_rates = [0.05, 0.1, 0.5, 0.9]
    for i, learning_rate in enumerate(learning_rates):
        ax = fig.add_subplot(2, 2, i + 1)
        for i, algorithm in enumerate(algorithms):
            clf = ensemble.AdaBoostClassifier(learning_rate=learning_rate,
                                              algorithm=algorithm)
            clf.fit(X_train, y_train)
            ## 绘图
            estimators_num = len(clf.estimators_)
            X = range(1, estimators_num + 1)
            ax.plot(list(X), list(clf.staged_score(X_train, y_train)),
                    label="%s:Traing score" % algorithms[i])
            ax.plot(list(X), list(clf.staged_score(X_test, y_test)),
                    label="%s:Testing score" % algorithms[i])
        ax.set_xlabel("estimator num")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_title("learing rate:%f" % learning_rate)
    fig.suptitle("AdaBoostClassifier")
    plt.show()


if __name__ == '__main__':
    corpus_dir = 'D:/UbunutWin/corpus/news_data/BQ20_seg'
    X_train, X_test, y_train, y_test = load_new_data(corpus_dir)

    do_AdaBoostClassifier(X_train, X_test, y_train, y_test)  # 调用 do_AdaBoostClassifier
    do_AdaBoostClassifier_learning_rate(X_train, X_test, y_train, y_test)  # 调用 do_AdaBoostClassifier_learning_rate
    do_AdaBoostClassifier_algorithm(X_train, X_test, y_train, y_test)  # 调用 do_AdaBoostClassifier_algorithm
