#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: classifyNewsBaseOnRF.py
@time: 2017/10/7 21:40
@desc: python3.6
"""
from sklearn import ensemble
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def load_new_data(corpus_dir):
    news = load_files(corpus_dir, encoding='utf-8')
    tfidf_vect = TfidfVectorizer()
    X = tfidf_vect.fit_transform(news.data)
    X_train, X_test, y_train, y_test = train_test_split(X, news.target, test_size=0.3, stratify=news.target)
    return X_train, X_test, y_train, y_test


def do_RandomForestClassifier(*data):
    '''
    测试 RandomForestClassifier 的用法

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    clf=ensemble.RandomForestClassifier(n_estimators = 100)
    clf.fit(X_train,y_train)
    print("Traing Score:%f"%clf.score(X_train,y_train))
    print("Testing Score:%f"%clf.score(X_test,y_test))
def do_RandomForestClassifier_num(*data):
    '''
    测试 RandomForestClassifier 的预测性能随 n_estimators 参数的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    nums=np.arange(1,100,step=2)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    testing_scores=[]
    training_scores=[]
    for num in nums:
        clf=ensemble.RandomForestClassifier(n_estimators=num)
        clf.fit(X_train,y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))
    ax.plot(nums,training_scores,label="Training Score")
    ax.plot(nums,testing_scores,label="Testing Score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0,1.05)
    plt.suptitle("RandomForestClassifier")
    plt.show()
def do_RandomForestClassifier_max_depth(*data):
    '''
    测试 RandomForestClassifier 的预测性能随 max_depth 参数的影响

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    maxdepths=range(1,20)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    testing_scores=[]
    training_scores=[]
    for max_depth in maxdepths:
        clf=ensemble.RandomForestClassifier(max_depth=max_depth)
        clf.fit(X_train,y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))
    ax.plot(maxdepths,training_scores,label="Training Score")
    ax.plot(maxdepths,testing_scores,label="Testing Score")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0,1.05)
    plt.suptitle("RandomForestClassifier")
    plt.show()
def do_RandomForestClassifier_max_features(*data):
    '''
    测试 RandomForestClassifier 的预测性能随 max_features 参数的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    max_features=np.linspace(0.01,1.0)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    testing_scores=[]
    training_scores=[]
    for max_feature in max_features:
        clf=ensemble.RandomForestClassifier(max_features=max_feature)
        clf.fit(X_train,y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))
    ax.plot(max_features,training_scores,label="Training Score")
    ax.plot(max_features,testing_scores,label="Testing Score")
    ax.set_xlabel("max_feature")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0,1.05)
    plt.suptitle("RandomForestClassifier")
    plt.show()

if __name__ == '__main__':
    corpus_dir = 'D:/UbunutWin/corpus/news_data/BQ20_seg'
    X_train, X_test, y_train, y_test = load_new_data(corpus_dir)
    do_RandomForestClassifier(X_train,X_test,y_train,y_test)  # 调用 do_RandomForestClassifier
    # do_RandomForestClassifier_num(X_train,X_test,y_train,y_test)  # 调用 do_RandomForestClassifier_num
    # do_RandomForestClassifier_max_depth(X_train,X_test,y_train,y_test)  # 调用 do_RandomForestClassifier_max_depth
    # do_RandomForestClassifier_max_features(X_train,X_test,y_train,y_test)  # 调用 do_RandomForestClassifier_max_features