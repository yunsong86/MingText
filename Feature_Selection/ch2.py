#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: ch2.py
@time: 2017/10/12 22:56
@desc: python3.6
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2

data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test')

y_train, y_test = data_train.target, data_test.target

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)
feature_names = vectorizer.get_feature_names()

ch2 = SelectKBest(chi2, k=200)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)

# keep selected feature names
feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]

print(len(feature_names))