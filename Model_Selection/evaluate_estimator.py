#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: evaluate_estimator.py
@time: 17-10-9 下午8:08
@desc: python3.6
"""
import logging
from sklearn import tree
from sklearn import svm
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from logging.handlers import RotatingFileHandler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

logging.root.setLevel(level=logging.INFO)
Rthandler = RotatingFileHandler(filename='evalute_estemators.log', mode='a', maxBytes=1 * 1024 * 1024, backupCount=2)
Rthandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
Rthandler.setFormatter(formatter)
logging.getLogger('').addHandler(Rthandler)

##################################################
# 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logger = logging.getLogger()

estimators = {}
estimators['gbdt'] = GradientBoostingClassifier(n_estimators=200, subsample=1.0)
# estimators['MultinomialNB'] = MultinomialNB()
# estimators['GaussianNB'] = GaussianNB()
# estimators['BernoulliNB'] = BernoulliNB()
# estimators['tree'] = tree.DecisionTreeClassifier()
# estimators['forest_100'] = RandomForestClassifier(n_estimators=100)
# estimators['svm_c_rbf'] = svm.SVC(probability=True)
# estimators['svm_c_linear'] = svm.SVC(kernel='linear', probability=True)
# estimators['svm_linear'] = svm.LinearSVC(C=0.1, multi_class='crammer_singer')
# estimators['svm_nusvc'] = svm.NuSVC(probability=True)
# estimators['lr'] = LogisticRegression()
# estimators['adboost'] = AdaBoostClassifier()
# estimators['bagging'] = BaggingClassifier()
# estimators['extratrees'] = ExtraTreesClassifier()
# estimators['sgd'] = SGDClassifier()
# estimators['ovo_svc_linear'] = OneVsOneClassifier(svm.LinearSVC(random_state=0))
# estimators['ovo_lr'] = OneVsOneClassifier(LogisticRegression(C=0.1))
# estimators['ovo_forest_10'] = OneVsOneClassifier(RandomForestClassifier(n_estimators=10, max_features=0.1))
# estimators['ovo_bayes'] = OneVsOneClassifier(MultinomialNB())
# estimators['ovr_srvc_linear'] = OneVsRestClassifier(svm.LinearSVC(random_state=0))
# estimators['ovr_lr'] = OneVsRestClassifier(LogisticRegression(C=0.1))
# estimators['ovr_forest_10'] = OneVsRestClassifier(RandomForestClassifier(n_estimators=10, max_features=0.1))
# estimators['ovr_bayes'] = OneVsRestClassifier(MultinomialNB())


def evalute_estemators(clf, estemator_name, X, y):
    if estemator_name == 'GBDT' or estemator_name == 'gdbt':
        X = X.toarray()
    clf.fit(X, y)
    score_evalute = cross_val_score(estimator=clf, X=X, y=y, cv=3, n_jobs=-1)
    logging.info("estemator name: " + str(estemator_name) + "\tevaluete score: " + str(score_evalute))


if __name__ == "__main__":
    # corpus_dir = '/mnt/hgfs/UbunutWin/corpus/twocategories'

    corpus_dir = '/mnt/hgfs/UbunutWin/corpus/news_data_seg'

    news = load_files(container_path=corpus_dir, encoding='utf-8')

    vectorizer = TfidfVectorizer(max_df=0.5)
    X = vectorizer.fit_transform(news.data)  # 文本转为词频矩阵
    y = news.target
    logging.info(X.shape)
    for name, clf in estimators.items():
        logging.info("*****%s*****",name)
        evalute_estemators(clf, name, X, y)
