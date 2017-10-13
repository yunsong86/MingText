#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: extract_company_person.py
@time: 2017/10/13 23:04
@desc: python3.6
"""
from NER.comom_resources import *
import copy

def seg_pos_ner(sent):
    word_cut = jieba.cut(str(sent))
    word_list = []
    for w in word_cut:
        word_list.append(w)
    postags_list = POSTTAGGER.postag(word_list)
    debug_postage_str = " ".join(postags_list)
    nertags_list = RECOGNIZER.recognize(word_list, postags_list)  #
    debug_str = " ".join(nertags_list)
    return word_list, postags_list, nertags_list

def brackets(content):
    res_dict = {}
    content = content.replace(u'(', u'（')
    content = content.replace(u')', u'）')
    regx = re.compile(r'\（[^\（\）]{2}\）')
    if regx.search(content):
        for w in regx.findall(content):
            h = content[content.find(w) - 2:content.find(w)]
            t = content[content.find(w) + len(w):content.find(w) + len(w) + 3]
            res_dict[w] = (h, t)

        content = re.sub(regx, '',content)
    return res_dict, content


def extract_campany(sent):
    word_list, postags_list, nertags_list = seg_pos_ner(sent)
    # 公司
    tmp = ''
    cpy_set = set()
    head_flag = False
    for i, nertags in enumerate(nertags_list):
        watch = word_list[i]
        if nertags[0] == 'B':
            head_flag = True
        if head_flag:
            tmp += word_list[i]
        if nertags[0] == 'E':
            tmp = tmp
            cpy_set.add(tmp)
            tmp = ''
            head_flag = False
    max_len = 3
    for cpy in cpy_set:
        if len(cpy) > max_len:
            max_len = len(cpy)
    cpy_list = list(cpy_set)
    for cpy in cpy_set:
        for suf in ENT_SUFFIX_LIST:
            regx = r'%s[\u4e00-\u9fa5]{1,%s}%s' % (cpy,  max_len,suf)
            if re.search(regx, sent):
                new_cpy = re.findall(regx, sent)[0]
                cpy_list.remove(cpy)
                cpy_list.append(new_cpy)
    return cpy_list

def modify_ent( bracket_dict, cpy_list):
    # 补回括号
    for key, value in bracket_dict.items():
        tmp = value[0] + value[1]
        for i, ent in enumerate(cpy_list):
            if tmp in ent:
                cpy_list[i] = ent.replace(tmp, value[0] + key + value[1])
    return  cpy_list

def extract_person(sent):
    word_list, postags_list, nertags_list = seg_pos_ner(sent)
    # 人
    flag = False
    for i, postags in enumerate(postags_list):
        debug_work_list = word_list[i]
        if postags == 'nh':
            pw = word_list[i]
            regx1 = r'^[\u4e00-\u9fa5]?%s'% pw
            regx = r'%s[\u4e00-\u9fa5]?$' % pw


if __name__ == '__main__':
    sent = '我在苏州（贝尔）塔数据技术有限公司江西销售分公司与江苏电力装备有限公司徐州加工厂'
    res_dict, new_sent =  brackets(sent)
    cpys = extract_campany(new_sent)
    cpys = modify_ent(res_dict, cpys)
    for c in cpys:
        print(c)