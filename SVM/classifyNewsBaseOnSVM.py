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
import matplotlib.pyplot as plt
import numpy as np
from sklearn import  svm


def load_new_data(corpus_dir):
    news = load_files(corpus_dir, encoding='utf-8')
    tfidf_vect = TfidfVectorizer()
    X = tfidf_vect.fit_transform(news.data)
    X_train, X_test, y_train, y_test = train_test_split(X, news.target, test_size=0.3, stratify=news.target)
    return X_train, X_test, y_train, y_test


def do_SVC_linear(*data):
    '''
    测试 SVC 的用法。这里使用的是最简单的线性核

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train, X_test, y_train, y_test = data
    cls = svm.SVC(kernel='linear')
    cls.fit(X_train, y_train)
    print('Coefficients:%s, intercept %s' % (cls.coef_, cls.intercept_))
    print('Score: %.2f' % cls.score(X_test, y_test))


def do_SVC_poly(*data):
    '''
    测试多项式核的 SVC 的预测性能随 degree、gamma、coef0 的影响.

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train, X_test, y_train, y_test = data
    fig = plt.figure()
    ### 测试 degree ####
    degrees = range(1, 20)
    train_scores = []
    test_scores = []
    for degree in degrees:
        cls = svm.SVC(kernel='poly', degree=degree)
        cls.fit(X_train, y_train)
        train_scores.append(cls.score(X_train, y_train))
        test_scores.append(cls.score(X_test, y_test))
    ax = fig.add_subplot(1, 3, 1)  # 一行三列
    ax.plot(degrees, train_scores, label="Training score ", marker='+')
    ax.plot(degrees, test_scores, label=" Testing  score ", marker='o')
    ax.set_title("SVC_poly_degree ")
    ax.set_xlabel("p")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", framealpha=0.5)

    ### 测试 gamma ，此时 degree 固定为 3####
    gammas = range(1, 20)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        cls = svm.SVC(kernel='poly', gamma=gamma, degree=3)
        cls.fit(X_train, y_train)
        train_scores.append(cls.score(X_train, y_train))
        test_scores.append(cls.score(X_test, y_test))
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(gammas, train_scores, label="Training score ", marker='+')
    ax.plot(gammas, test_scores, label=" Testing  score ", marker='o')
    ax.set_title("SVC_poly_gamma ")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", framealpha=0.5)
    ### 测试 r ，此时 gamma固定为10 ， degree 固定为 3######
    rs = range(0, 20)
    train_scores = []
    test_scores = []
    for r in rs:
        cls = svm.SVC(kernel='poly', gamma=10, degree=3, coef0=r)
        cls.fit(X_train, y_train)
        train_scores.append(cls.score(X_train, y_train))
        test_scores.append(cls.score(X_test, y_test))
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(rs, train_scores, label="Training score ", marker='+')
    ax.plot(rs, test_scores, label=" Testing  score ", marker='o')
    ax.set_title("SVC_poly_r ")
    ax.set_xlabel(r"r")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", framealpha=0.5)
    plt.show()


def do_SVC_rbf(*data):
    '''
    测试 高斯核的 SVC 的预测性能随 gamma 参数的影响

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train, X_test, y_train, y_test = data
    gammas = range(1, 20)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        cls = svm.SVC(kernel='rbf', gamma=gamma)
        cls.fit(X_train, y_train)
        train_scores.append(cls.score(X_train, y_train))
        test_scores.append(cls.score(X_test, y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(gammas, train_scores, label="Training score ", marker='+')
    ax.plot(gammas, test_scores, label=" Testing  score ", marker='o')
    ax.set_title("SVC_rbf")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", framealpha=0.5)
    plt.show()


def do_SVC_sigmoid(*data):
    '''
    测试 sigmoid 核的 SVC 的预测性能随 gamma、coef0 的影响.

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train, X_test, y_train, y_test = data
    fig = plt.figure()

    ### 测试 gamma ，固定 coef0 为 0 ####
    gammas = np.logspace(-2, 1)
    train_scores = []
    test_scores = []

    for gamma in gammas:
        cls = svm.SVC(kernel='sigmoid', gamma=gamma, coef0=0)
        cls.fit(X_train, y_train)
        train_scores.append(cls.score(X_train, y_train))
        test_scores.append(cls.score(X_test, y_test))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(gammas, train_scores, label="Training score ", marker='+')
    ax.plot(gammas, test_scores, label=" Testing  score ", marker='o')
    ax.set_title("SVC_sigmoid_gamma ")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", framealpha=0.5)
    ### 测试 r，固定 gamma 为 0.01 ######
    rs = np.linspace(0, 5)
    train_scores = []
    test_scores = []

    for r in rs:
        cls = svm.SVC(kernel='sigmoid', coef0=r, gamma=0.01)
        cls.fit(X_train, y_train)
        train_scores.append(cls.score(X_train, y_train))
        test_scores.append(cls.score(X_test, y_test))
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(rs, train_scores, label="Training score ", marker='+')
    ax.plot(rs, test_scores, label=" Testing  score ", marker='o')
    ax.set_title("SVC_sigmoid_r ")
    ax.set_xlabel(r"r")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", framealpha=0.5)
    plt.show()


def do_LinearSVC(*data):
    '''
    测试 LinearSVC 的用法

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train, X_test, y_train, y_test = data
    cls = svm.LinearSVC()
    cls.fit(X_train, y_train)
    print('Coefficients:%s, intercept %s' % (cls.coef_, cls.intercept_))
    print('Score: %.2f' % cls.score(X_test, y_test))


def do_LinearSVC_loss(*data):
    '''
    测试 LinearSVC 的预测性能随损失函数的影响

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train, X_test, y_train, y_test = data
    losses = ['hinge', 'squared_hinge']
    for loss in losses:
        cls = svm.LinearSVC(loss=loss)
        cls.fit(X_train, y_train)
        print("Loss:%f" % loss)
        print('Coefficients:%s, intercept %s' % (cls.coef_, cls.intercept_))
        print('Score: %.2f' % cls.score(X_test, y_test))


def do_LinearSVC_L12(*data):
    '''
    测试 LinearSVC 的预测性能随正则化形式的影响

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train, X_test, y_train, y_test = data
    L12 = ['l1', 'l2']
    for p in L12:
        cls = svm.LinearSVC(penalty=p, dual=False)
        cls.fit(X_train, y_train)
        print("penalty:%s" % p)
        print('Coefficients:%s, intercept %s' % (cls.coef_, cls.intercept_))
        print('Score: %.2f' % cls.score(X_test, y_test))


def do_LinearSVC_C(*data):
    '''
    测试 LinearSVC 的预测性能随参数 C 的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:   None
    '''
    X_train, X_test, y_train, y_test = data
    Cs = np.logspace(-2, 1)
    train_scores = []
    test_scores = []
    for C in Cs:
        cls = svm.LinearSVC(C=C)
        cls.fit(X_train, y_train)
        train_scores.append(cls.score(X_train, y_train))
        test_scores.append(cls.score(X_test, y_test))

    ## 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Cs, train_scores, label="Traing score")
    ax.plot(Cs, test_scores, label="Testing score")
    ax.set_xlabel(r"C")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("LinearSVC")
    ax.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    corpus_dir = 'D:/UbunutWin/corpus/news_data/BQ20_seg'
    X_train, X_test, y_train, y_test = load_new_data(corpus_dir)

    # do_SVC_linear(X_train, X_test, y_train, y_test)  # 调用 do_SVC_linear
    do_SVC_poly(X_train,X_test,y_train,y_test) # 调用 do_SVC_poly
    do_SVC_rbf(X_train,X_test,y_train,y_test) # 调用 do_SVC_rbf
    do_SVC_sigmoid(X_train,X_test,y_train,y_test) # do_SVC_sigmoid do_SVC_linear
    #
    do_LinearSVC(X_train, X_test, y_train, y_test)  # 调用 do_LinearSVC
    do_LinearSVC_loss(X_train, X_test, y_train, y_test)  # 调用 do_LinearSVC_loss
    do_LinearSVC_L12(X_train, X_test, y_train, y_test)  # 调用 do_LinearSVC_L12
    do_LinearSVC_C(X_train, X_test, y_train, y_test)  # 调用 do_LinearSVC_C
