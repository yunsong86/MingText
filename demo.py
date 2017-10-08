#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: demo.py
@time: 2017/10/8 2:05
@desc: python3.6
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [' played Duke  game', 'played lost  basketball game', 'I   sandwich']
vectorizer = CountVectorizer()

corpus_vect_count = vectorizer.fit_transform(corpus)

corpusTotoken = vectorizer.fit_transform(corpus).todense()

print(corpusTotoken)

tfidfvector = TfidfVectorizer()

tfidfvect = tfidfvector.fit_transform(corpus)
print(tfidfvect)
tfidfvect2 = tfidfvector.fit_transform(corpus).todense()
print(tfidfvect2)
print('ddd')

# 语料
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
# 将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
# 计算个词语出现的次数
X = vectorizer.fit_transform(corpus)
# 获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
print(word)
# 查看词频结果
print(X.toarray())
print(X.todense())
from sklearn.feature_extraction.text import TfidfTransformer

# 类调用
transformer = TfidfTransformer()
print(transformer)
# 将词频矩阵X统计成TF-IDF值
tfidf = transformer.fit_transform(X)
# 查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
print(tfidf.toarray())
