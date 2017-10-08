#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: classifyNewsBaseOnLR.py
@time: 2017/10/7 23:31
@desc: python3.6
"""
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def load_new_data(corpus_dir):
    news = load_files(corpus_dir, encoding='utf-8')
    tfidf_vect = TfidfVectorizer()
    X = tfidf_vect.fit_transform(news.data)
    X_train, X_test, y_train, y_test = train_test_split(X, news.target, test_size=0.3, stratify=news.target)
    return X_train, X_test, y_train, y_test

def do_LogisticRegression(*data):
    '''
    测试 LogisticRegression 的用法

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    regr = LogisticRegression()
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %s'%(regr.coef_,regr.intercept_))
    print('Score: %.2f' % regr.score(X_test, y_test))
def do_LogisticRegression_multinomial(*data):
    '''
    测试 LogisticRegression 的预测性能随 multi_class 参数的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    regr = LogisticRegression(multi_class='multinomial',solver='lbfgs')
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %s'%(regr.coef_,regr.intercept_))
    print('Score: %.2f' % regr.score(X_test, y_test))
def do_LogisticRegression_C(*data):
    '''
    测试 LogisticRegression 的预测性能随  C  参数的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    Cs=np.logspace(-2,4,num=100)
    scores=[]
    for C in Cs:
        regr = LogisticRegression(C=C)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(Cs,scores)
    ax.set_xlabel(r"C")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("LogisticRegression")
    plt.show()

if __name__=='__main__':
    corpus_dir = 'D:/UbunutWin/corpus/news_data/BQ20_seg'
    X_train, X_test, y_train, y_test = load_new_data(corpus_dir)
    do_LogisticRegression(X_train,X_test,y_train,y_test) # 调用  do_LogisticRegression
    # do_LogisticRegression_multinomial(X_train,X_test,y_train,y_test) # 调用  do_LogisticRegression_multinomial
    # do_LogisticRegression_C(X_train,X_test,y_train,y_test) # 调用  do_LogisticRegression_C