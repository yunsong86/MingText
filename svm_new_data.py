#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: svm_new_data.py
@time: 17-9-30 下午3:00
@desc: python3.6
"""
import pickle
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

file_obj = open('/mnt/hgfs/UbunutWin/corpus/news_data_model/tfidfwordbag.dat', 'rb')
tfidf_wordbag_Bunch = pickle.load(file_obj)
file_obj.close()

X_train, X_test, y_train, y_test = train_test_split(tfidf_wordbag_Bunch.tfidf, tfidf_wordbag_Bunch.label, test_size=0.2,
                                                    random_state=0, stratify=tfidf_wordbag_Bunch.label)

svclf = LinearSVC(penalty="l1", dual=False, tol=1e-4, class_weight="balanced" )
# 训练分类器
svclf.fit(X_train, y_train)
# 预测分类结果
predicted = svclf.predict(X_test)

print(classification_report(y_test, predicted, target_names=tfidf_wordbag_Bunch.target_name))
