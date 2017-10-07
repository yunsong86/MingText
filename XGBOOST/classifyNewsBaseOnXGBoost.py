#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: classifyNewsBaseOnXGBoost.py
@time: 2017/10/7 21:44
@desc: python3.6
"""

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.model_selection import train_test_split
