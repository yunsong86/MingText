#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: tfidf.py
@time: 2017/10/11 22:51
@desc: python3.6
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

docs = ['北京 北京 北京   北京 上海  南京 ',
        '上海 上海 上海 上海  上海 上海 南京 北京 北京',
        '北京,上海',
        '北京 杭州'

        ]

vector = CountVectorizer(min_df=1,max_df=4)
tfidf_vect = vector.fit_transform(docs)
wordlist = vector.get_feature_names()  # 获取词袋模型中的所有词
# tf-idf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
weightlist = tfidf_vect.toarray()
# 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
for i in range(len(weightlist)):
    print("-------这里输出第", i, "类文本的词语tf-idf权重------")
    for j in range(len(wordlist)):
        print(wordlist[j], weightlist[i][j])
print(tfidf_vect.toarray())
print()



