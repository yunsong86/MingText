#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@license: (C) Copyright 2013-2017,Bertadata Corporation Limited.
@contact: yunsong@bertadata.com
@file: model_read_only.py
@time: 2017/7/28 7:32
@desc: python2
"""
import sys
import gensim
from datetime import datetime

reload(sys)
sys.setdefaultencoding('utf-8')

starttime = datetime.now()

model = gensim.models.Word2Vec.load("E:\ysworkspace\process_sougou_new\w2v_model\wiki.sogou.250.model")

model.init_sims(replace=True)

for (k, v) in model.wv.most_similar(positive=[u'男人', u'儿子'], negative=[u'女儿']):
    print  k, v
print '-------------'
print  model.wv.similarity(u'男人', u'女人')

print '-------------'
for (k, v) in model.wv.most_similar(u'男人', u'女人'):
    print  k, v
print model.doesnt_match(u"我 打 你".split())

endtime = datetime.now()
print (endtime-starttime).seconds
