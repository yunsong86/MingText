#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: Log.py
@time: 9/28/17 1:02 AM
@desc: python3.6
"""
import logging
from logging.handlers import RotatingFileHandler

logging.root.setLevel(level=logging.INFO)
Rthandler = RotatingFileHandler(filename='mingtext.log', mode='a', maxBytes=1 * 1024 * 1024, backupCount=2)
Rthandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
Rthandler.setFormatter(formatter)
logging.getLogger('').addHandler(Rthandler)

##################################################
#定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


logger = logging.getLogger()
