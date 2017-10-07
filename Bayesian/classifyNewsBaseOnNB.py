#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: classifyNewsBaseOnNB.py
@time: 2017/10/7 20:53
@desc: python3.6
"""



from sklearn import datasets, model_selection, naive_bayes
import matplotlib.pyplot as plt
from Bayesian.gaussianNB import do_GaussianNB
from Bayesian.multinomialNB import do_MultinomialNB, do_MultinomialNB_alpha
from Bayesian.bernoulliNB import do_BernoulliNB, do_BernoulliNB_alpha, do_BernoulliNB_binarize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.datasets import load_files


def load_data():
    '''
    加载用于分类问题的数据集。这里使用 scikit-learn 自带的 digits 数据集

    :return: 一个元组，用于分类问题。元组元素依次为：训练样本集、测试样本集、训练样本集对应的标记、测试样本集对应的标记
    '''
    digits = datasets.load_digits()  # 加载 scikit-learn 自带的 digits 数据集
    return model_selection.train_test_split(digits.data, digits.target,
                                             test_size=0.25, random_state=0,
                                             stratify=digits.target)  # 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4
def load_new_data(corpus_dir):
    news = load_files(corpus_dir, encoding='utf-8')
    tfidf_vect = TfidfVectorizer()
    X = tfidf_vect.fit_transform(news.data)
    X_train, X_test, y_train, y_test = train_test_split(X, news.target, test_size=0.3, stratify=news.target)
    return X_train, X_test, y_train, y_test

def show_digits():
    '''
    绘制 digits 数据集。这里只是绘制数据集中前 25 个样本的图片。

    :return: None
    '''
    digits = datasets.load_digits()
    fig = plt.figure()
    print("vector from images 0:", digits.data[0])
    for i in range(25):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    corpus_dir = 'D:/UbunutWin/corpus/news_data/BQ20_seg'
    X_train, X_test, y_train, y_test = load_new_data(corpus_dir)
    # do_MultinomialNB(X_train, X_test, y_train, y_test)  # 调用 test_MultinomialNB
    # do_MultinomialNB_alpha(X_train, X_test, y_train, y_test)  # 调用 test_MultinomialNB_alpha
    do_BernoulliNB(X_train, X_test, y_train, y_test)  # 调用 test_BernoulliNB
    do_BernoulliNB_alpha(X_train, X_test, y_train, y_test)  # 调用 test_BernoulliNB_alpha
