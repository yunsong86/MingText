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

logger = logging.getLogger()


