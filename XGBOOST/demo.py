#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: demo.py
@time: 2017/10/8 11:05
@desc: python3.6
"""

import xgboost as xgb
import csv
import jieba

jieba.load_userdict('wordDict.txt')
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# 读取训练集
def readtrain():
    with open('Train.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row for row in reader]
    content_train = [i[1] for i in column1[1:]]  # 第一列为文本内容，并去除列名
    opinion_train = [i[2] for i in column1[1:]]  # 第二列为类别，并去除列名
    print('训练集有 %s 条句子' % len(content_train))
    train = [content_train, opinion_train]
    return train


# 将utf8的列表转换成unicode
def changeListCode(b):
    a = []
    for i in b:
        a.append(i.decode('utf8'))
    return a


# 对列表进行分词并用空格连接
def segmentWord(cont):
    c = []
    for i in cont:
        a = list(jieba.cut(i))
        b = " ".join(a)
        c.append(b)
    return c


# 类别用数字表示：pos:2,neu:1,neg:0
def transLabel(labels):
    for i in range(len(labels)):
        if labels[i] == 'pos':
            labels[i] = 2
        elif labels[i] == 'neu':
            labels[i] = 1
        elif labels[i] == 'neg':
            labels[i] = 0
        else:
            print("label无效：", labels[i])
    return labels


train = readtrain()
content = segmentWord(train[0])
opinion = transLabel(train[1])  # 需要用数字表示类别
opinion = np.array(opinion)  # 需要numpy格式

train_content = content[:7000]
train_opinion = opinion[:7000]
test_content = content[7000:]
test_opinion = opinion[7000:]

vectorizer = CountVectorizer()
tfidftransformer = TfidfTransformer()
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train_content))
weight = tfidf.toarray()
print(tfidf.shape)
test_tfidf = tfidftransformer.transform(vectorizer.transform(test_content))
test_weight = test_tfidf.toarray()
print(test_weight.shape)

dtrain = xgb.DMatrix(weight, label=train_opinion)
dtest = xgb.DMatrix(test_weight, label=test_opinion)  # label可以不要，此处需要是为了测试效果
param = {'max_depth': 6, 'eta': 0.5, 'eval_metric': 'merror', 'silent': 1, 'objective': 'multi:softmax',
         'num_class': 3}  # 参数
evallist = [(dtrain, 'train'), (dtest, 'test')]  # 这步可以不要，用于测试效果
num_round = 50  # 循环次数
bst = xgb.train(param, dtrain, num_round, evallist)
preds = bst.predict(dtest)
