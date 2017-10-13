# -*- coding: utf-8 -*-

"""
@author: PANYUNSONG
@license: (C) Copyright 2013-2017,Bertadata Corporation Limited.
@contact: yunsong@bertadata.com
@file: com_resource.py
@time: 2017/8/16 9:34
@desc: python2
"""
import sys
import re
import jieba
from pyltp import *
import codecs


PATH_RESOURCES = 'D:/UbuntuWin/resources/'

# logger = Nlog.nlogger(LOG_CONFIG)


_SEG_MODEL_FILE = PATH_RESOURCES + 'ltp_data_v3.4.0/cws.model'
_NER_MODEL_FILE = PATH_RESOURCES + 'ltp_data_v3.4.0/ner.model'
_POS_MODEL_FILE = PATH_RESOURCES + 'ltp_data_v3.4.0/pos.model'
_JIEBA_USER_DICT = PATH_RESOURCES + 'jieba_data/jiebauserdict.txt'
jieba.load_userdict(_JIEBA_USER_DICT)

POSTTAGGER = Postagger()
SEGMENTOR = Segmentor()
RECOGNIZER = NamedEntityRecognizer()

POSTTAGGER.load(_POS_MODEL_FILE)
SEGMENTOR.load(_SEG_MODEL_FILE)
RECOGNIZER.load(_NER_MODEL_FILE)





# 机构后缀
ENT_SUFFIX_LIST = ['分公司', u'站', u'队', u'日报社', u'事务所', u'供电段', u'销售部', u'工务段', u'铁路局', u'百货行',
                   u'经营部', u'维修部', u'工程处', u'工程队', u'工程部', u'货运部', u'经销处', u'服务中心',
                   u'服务部', u'服务队', u'加盟店', u'分店', u'专卖店', u'配件厂', u'修造厂', u'材料厂',
                   u'养殖场', u'采石厂', u'修理厂', u'木材厂', u'加工厂', u'食品厂', u'养殖厂', u'配件厂',u'合作联社',
                   u'制品厂', u'合作社', u'事务所', u'管理局', u'商店', u'宾馆', u'酒楼', u'酒店', u'租赁部', u'分行',u'支行']


# 公司名
ENT_NAME = [u'长沙市开福区永益不干胶加工经营部',
            u'中国邮政储蓄银行股份有限公司尚志市支行',
            u'中国民生银行股份有限公司北京分行',
            u'钦州市区农村信用合作联社',
            u'天津市南开中小企业非融资性信用担保有限公司',
            u'黄山永信非融资性担保有限公司',
            u'赣州开发区潭东镇振兴钢管租赁部',
            u'九三粮油工业集团有限公司']
