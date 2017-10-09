# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yun Song 
# Copyrigh 2017
import sys
import jieba

reload(sys)
sys.setdefaultencoding('utf-8')

__author__ = "yunsong"

import codecs
import jieba

infile = 'C:\Users\user\Desktop\sougounewcorpus.txt'
outfile = 'C:\Users\user\Desktop\sougounewcorpus.seg'
descsFile = codecs.open(infile, 'rb', encoding='utf-8')
i = 0
with codecs.open(outfile, 'w', encoding='utf-8') as f:
    for line in descsFile:
        i += 1
        if i % 10000 == 0:
            print(i)
        line = line.strip()
        words = jieba.cut(line)
        for word in words:
            f.write(word + ' ')
        f.write('\n')
