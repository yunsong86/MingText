# -*- coding: utf-8 -*-
"""
    集成学习
    ~~~~~~~~~~~~~~~~

    GradientBoostingClassifier

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, cross_validation, ensemble


def load_data_classification():
    '''
    加载用于分类问题的数据集

    :return: 一个元组，用于分类问题。元组元素依次为：训练样本集、测试样本集、训练样本集对应的标记、测试样本集对应的标记
    '''
    digits = datasets.load_digits()  # 使用 scikit-learn 自带的 digits 数据集
    return cross_validation.train_test_split(digits.data, digits.target,
                                             test_size=0.25, random_state=0,
                                             stratify=digits.target)  # 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4


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
    X_train, X_test, y_train, y_test = load_data_classification()  # 获取分类数据
    do_GradientBoostingClassifier(X_train, X_test, y_train, y_test)  # 调用 do_GradientBoostingClassifier
    # do_GradientBoostingClassifier_num(X_train,X_test,y_train,y_test) # 调用 do_GradientBoostingClassifier_num
    # do_GradientBoostingClassifier_maxdepth(X_train,X_test,y_train,y_test) # 调用 do_GradientBoostingClassifier_maxdepth
    # do_GradientBoostingClassifier_learning(X_train,X_test,y_train,y_test) # 调用 do_GradientBoostingClassifier_learning
    # do_GradientBoostingClassifier_subsample(X_train,X_test,y_train,y_test) # 调用 do_GradientBoostingClassifier_subsample
    # do_GradientBoostingClassifier_max_features(X_train,X_test,y_train,y_test) # 调用 do_GradientBoostingClassifier_max_features
