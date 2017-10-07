#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: classifyNewsBaseOnDT.py
@time: 2017/10/7 21:25
@desc: python3.6
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def do_DecisionTreeClassifier(*data):
    '''
    测试 DecisionTreeClassifier 的用法

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train, X_test, y_train, y_test = data
    clf = DecisionTreeClassifier(class_weight='balanced')
    clf.fit(X_train, y_train)

    print("Training score:%f" % (clf.score(X_train, y_train)))
    print("Testing score:%f" % (clf.score(X_test, y_test)))


def do_DecisionTreeClassifier_criterion(*data):
    '''
    测试 DecisionTreeClassifier 的预测性能随 criterion 参数的影响

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train, X_test, y_train, y_test = data
    criterions = ['gini', 'entropy']
    for criterion in criterions:
        clf = DecisionTreeClassifier(criterion=criterion, class_weight='balanced')
        clf.fit(X_train, y_train)
        print("criterion:%s" % criterion)
        print("Training score:%f" % (clf.score(X_train, y_train)))
        print("Testing score:%f" % (clf.score(X_test, y_test)))


def do_DecisionTreeClassifier_splitter(*data):
    '''
    测试 DecisionTreeClassifier 的预测性能随划分类型的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train, X_test, y_train, y_test = data
    splitters = ['best', 'random']
    for splitter in splitters:
        clf = DecisionTreeClassifier(splitter=splitter, class_weight='balanced')
        clf.fit(X_train, y_train)
        print("splitter:%s" % splitter)
        print("Training score:%f" % (clf.score(X_train, y_train)))
        print("Testing score:%f" % (clf.score(X_test, y_test)))


def do_DecisionTreeClassifier_depth(*data, maxdepth):
    '''
    测试 DecisionTreeClassifier 的预测性能随 max_depth 参数的影响

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :param maxdepth: 一个整数，用于 DecisionTreeClassifier 的 max_depth 参数
    :return:  None
    '''
    X_train, X_test, y_train, y_test = data
    depths = np.arange(1, maxdepth)
    training_scores = []
    testing_scores = []
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, class_weight='balanced')
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train, y_train))
        testing_scores.append(clf.score(X_test, y_test))

    ## 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(depths, training_scores, label="traing score", marker='o')
    ax.plot(depths, testing_scores, label="testing score", marker='*')
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Classification")
    ax.legend(framealpha=0.5, loc='best')
    plt.show()


def load_new_data(corpus_dir):
    news = load_files(corpus_dir, encoding='utf-8')
    tfidf_vect = TfidfVectorizer()
    X = tfidf_vect.fit_transform(news.data)
    X_train, X_test, y_train, y_test = train_test_split(X, news.target, test_size=0.3, stratify=news.target)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    corpus_dir = 'D:/UbunutWin/corpus/news_data/BQ20_seg'
    X_train, X_test, y_train, y_test = load_new_data(corpus_dir)
    # do_DecisionTreeClassifier(X_train, X_test, y_train, y_test)  # 调用 do_DecisionTreeClassifier
    do_DecisionTreeClassifier_criterion(X_train, X_test, y_train, y_test)  # 调用 do_DecisionTreeClassifier_criterion
    do_DecisionTreeClassifier_splitter(X_train, X_test, y_train, y_test)  # 调用 do_DecisionTreeClassifier_splitter
    do_DecisionTreeClassifier_depth(X_train, X_test, y_train, y_test,
                                    maxdepth=100)  # 调用 do_DecisionTreeClassifier_depth
