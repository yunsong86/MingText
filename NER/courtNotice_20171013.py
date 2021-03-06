#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@license: (C) Copyright 2013-2017,Bertadata Corporation Limited.
@contact: yunsong@bertadata.com
@file: countNotice3nd.py
@time: 2017/9/20 14:10
@desc: python2
"""
import sys
import re
import json
from NER.comom_resources import *
import copy




def check_content(content):
    first_role = ''
    second_role = ''
    content = content.decode('utf-8')
    if u'执行' in content:
        first_role = u'申请执行人'
        second_role = u'被执行人'
    elif u'上诉人' in content:
        first_role = u'上诉人'
        second_role = u'被上诉人'
    elif u'申请人' in content and u'原告' not in content:
        first_role = u'申请人'
        second_role = u'被申请人'
    else:
        first_role = u'原告'
        second_role = u'被告'

    return first_role, second_role


# 分割
def split_content(content):
    # regx = ur'\（[\u4e00-\u9fa5]+[:：]?\d+\）'
    # if re.search(regx, content):
    #     tmp = re.findall(regx, content)[0]
    #     tmp2 = tmp.replace(u'：', '').replace(u':', '')
    #     content = content.replace(tmp, tmp2)
    #
    # rule_content = ''
    # ner_content = ''
    # if u'：' in content or u':' in content:
    #     split_content_list = re.split(u'[:：]', content)
    #     rule_content = split_content_list[0]
    #     ner_content = split_content_list[1]
    # else:
    #     ner_content = content
    # return rule_content, ner_content

    # --------------------------------------------
    # content = content.replace(u':', u'：')
    # # 去掉（：12345）这种特例
    # regx = ur'\（[\u4e00-\u9fa5]+：\d+\）'
    # if re.search(regx, content):
    #     tmp = re.findall(regx, content)[0]
    #     tmp2 = tmp.replace(u'：', '')
    #     content = content.replace(tmp, tmp2)

    rule_content = ''
    ner_content = ''
    find_content = content.replace(u':', u'：')
    find_content = find_content.replace(u'(', u'（')
    find_content = find_content.replace(u')', u'）')
    find_content = re.sub(ur'\（[^\（\）]{7,}\）', '', find_content)
    x1 = find_content.find(u'：')
    x2 = find_content.find(u'。')

    # 去掉（：12345）这种特例
    content = content.replace(u':', u'：')
    regx = ur'\（[\u4e00-\u9fa5]+：\d+\）'
    if re.search(regx, content):
        tmp = re.findall(regx, content)[0]
        tmp2 = tmp.replace(u'：', '')
        content = content.replace(tmp, tmp2)

    if x1 != -1 and x1 < x2:
        split_content_list = re.split(u'：', content)
        rule_content = split_content_list[0]
        ner_content = split_content_list[1]
    else:
        ner_content = content
    return rule_content, ner_content


def split_content2nd(content):
    content = content.replace(u':', u'：')
    # 去掉（：12345）这种特例
    regx = ur'\（[\u4e00-\u9fa5]+：\d+\）'
    if re.search(regx, content):
        tmp = re.findall(regx, content)[0]
        tmp2 = tmp.replace(u'：', '')
        content = content.replace(tmp, tmp2)

    rule_content = ''
    ner_content = ''
    x1 = content.find(u'：')
    x2 = content.find(u'。')
    if x1 != -1 and x1 < x2:
        split_content_list = re.split(u'：', content)
        rule_content = split_content_list[0]
        ner_content = split_content_list[1]
    else:
        ner_content = content
    return rule_content, ner_content


# 预处理
def preprocess(content):
    content = content.decode('utf-8')
    content = content.replace(u'原告', u'')
    content = content.replace(u'被告', u'')
    content = content.replace(u'被执行人', u'')
    content = content.replace(u'执行人', u'')
    content = content.replace(u'被上诉人', u'')
    content = content.replace(u'上诉人', u'')

    content = content.replace(u'被申请人', u'')
    content = content.replace(u'申请人', u'')

    content = content.replace(u'被告', u'')
    content = content.replace(u'本院审理的', u'')
    content = content.replace(u'本院受理', u'')
    content = content.replace(u'本院对', u'')
    content = content.replace(u'被执行人', u'')
    content = content.replace(u'诉', u'起诉')
    content = re.sub(u'\([^\(\)]{2}\)', '', content)
    content = re.sub(u'\（[^\（\）]{2}\）', '', content)
    return content


# 截取
def truncation_by_rule(content, second_role):
    entity_set = set()
    entity_role_list = []
    if content:
        for r in re.split(u'[、，,；;]', content):
            r = re.sub(u'\([^\(\)]+\)$', '', r)
            r = re.sub(u'\（[^\（\）]+\）$', '', r)

            entity_set.add(r)
            entity_role_list.append((r, second_role))

    # 规则再过滤

    return entity_set, entity_role_list


def clean_entity(entity):
    pass


def ner_entity(content, truncation_entity_set):
    ner_entity_set = set()
    if content:

        for name in ENT_NAME:
            if name in content:
                ner_entity_set.add(name)

        content = preprocess(content)
        for sent in re.split(ur'[；。，;,\r\n]', content):
            # if u'法院' in sent: continue
            if sent:

                word_cut = jieba.cut(str(sent))
                word_list = []
                for w in word_cut:
                    word_list.append(w.encode('utf-8'))
                postags_list = POSTTAGGER.postag(word_list)
                debug_postage_str = " ".join(postags_list)
                nertags_list = RECOGNIZER.recognize(word_list, postags_list)  #
                debug_str = " ".join(nertags_list)
                # 公司
                tmp = ''
                head_flag = False
                for i, nertags in enumerate(nertags_list):
                    watch = word_list[i]
                    if nertags[0] == 'B':
                        head_flag = True
                    if head_flag:
                        tmp += word_list[i]
                    if nertags[0] == 'E':
                        tmp = tmp.decode('utf-8')
                        if tmp not in truncation_entity_set:
                            ner_entity_set.add(tmp)
                        tmp = ''
                        head_flag = False

                ner_person(postags_list, word_list, ner_entity_set, truncation_entity_set)


    return ner_entity_set


def ner_person(postags_list, word_list, ner_entity_set, truncation_entity_set):
    # 人
    flag = False
    for i, postags in enumerate(postags_list):
        debug_work_list = word_list[i]
        if flag:
            flag = False
            continue

        if postags == 'nh':
            flag_person = True
            for ent in ner_entity_set:
                if word_list[i] in ent:
                    flag_person = False
            for ent in truncation_entity_set:
                if word_list[i] in ent:
                    flag_person = False
            if flag_person:
                # 修正各种特例
                # X  XX
                if len(word_list[i].decode('utf-8')) == 1 and i + 1 < len(word_list):
                    if (word_list[i] + word_list[i + 1]) not in truncation_entity_set:
                        ner_entity_set.add(word_list[i] + word_list[i + 1])
                        flag = True
                # XX X
                elif len(word_list[i].decode('utf-8')) == 2 and i + 2 == len(word_list) and len(
                        word_list[i + 1].decode('utf-8')) == 1:
                    if (word_list[i] + word_list[i + 1]) not in truncation_entity_set:
                        ner_entity_set.add(word_list[i] + word_list[i + 1])
                        flag = True
                elif i + 1 < len(word_list) and postags_list[i + 1] == 'nh':
                    if (word_list[i] + word_list[i + 1]) not in truncation_entity_set:
                        ner_entity_set.add(word_list[i] + word_list[i + 1])
                        flag = True
                elif i -1>= 0 and postags_list[i - 1] == 'nh':
                    if (word_list[i-1] + word_list[i]) not in truncation_entity_set:
                        ner_entity_set.add(word_list[i-1] + word_list[i ])
                        flag = True


                else:
                    if (word_list[i]) not in truncation_entity_set:
                        ner_entity_set.add(word_list[i])
                        flag = False


def brackets(content):
    res_dict = {}
    content = content.replace(u'(', u'（')
    content = content.replace(u')', u'）')
    regx = re.compile(ur'\（[^\（\）]{2}\）')
    if regx.search(content):
        for w in regx.findall(content):
            h = content[content.find(w) - 2:content.find(w)]
            t = content[content.find(w) + len(w):content.find(w) + len(w) + 3]
            res_dict[w] = (h, t)
    return res_dict


def modify_ent(content, bracket_dict, ner_entity_set):
    content = content.replace(u'(', u'（')
    content = content.replace(u')', u'）')
    entity_list = list(ner_entity_set)
    # 补回括号
    for key, value in bracket_dict.items():
        tmp = value[0] + value[1]
        for i, ent in enumerate(entity_list):
            if tmp in ent:
                entity_list[i] = ent.replace(tmp, value[0] + key + value[1])

    # 原告牛艳霞
    for i, ent in enumerate(entity_list):
        role_list = [u'原告', u'被告', u'被上诉人', u'被申请人', u'上诉人', u'申请人', u'被执行人', u'执行人']
        for role in role_list:
            regx = ur'%s[\u4e00-\u9fa5]{1}%s' % (role, ent)
            if re.search(regx, content):
                tmp = re.findall(regx, content)[0]
                tmp = tmp.replace(role, '')
                entity_list[i] = tmp
                break

    # 词典匹配修正
    for i, ent in enumerate(entity_list):
        for name in ENT_NAME:
            if name in content and ent in name:
                entity_list[i] = name
    entity_list = list(set(entity_list))
    return entity_list


# 第二次升级版本
def modify_ent2nd(content, entity_list):
    # 机构后缀
    suffix_list = [u'分公司', u'支行', u'分行', u'分公司', u'站', u'队', u'日报社', u'事务所', u'供电段', u'销售部',
                   u'工务段', u'铁路局', u'百货行', u'经营部', u'维修部', u'工程处', u'工程队', u'工程部', u'货运部', u'经销处', u'服务中心',
                   u'服务部', u'服务队', u'加盟店', u'分店', u'专卖店', u'配件厂', u'修造厂', u'材料厂', u'养殖场', u'采石厂', u'修理厂', u'木材厂',
                   u'加工厂', u'食品厂', u'养殖厂', u'配件厂', u'制品厂', u'合作社', u'事务所', u'管理局', u'商店', u'宾馆', u'酒楼', u'酒店', ]
    modify_content = content.replace(u'诉', u'。').replace(u'和', u'。').replace(u'与', u'。')
    sentences_list = re.split(ur'[，。、,.]', modify_content)
    modify_entity_list = []
    entity_set = set()
    del_entity_list = []

    for ent in entity_list:
        flag = True
        flag2 = True
        for sent in sentences_list:
            if not flag: break

            regx2 = ur'%s[\u4e00-\u9fa5]+人民法院' % (ent)
            if re.search(regx2, sent):
                del_entity_list.append(ent)
                flag2 = False
                break

            for suf in suffix_list:
                regx = ur'%s[\u4e00-\u9fa5]+%s' % (ent, suf)
                if re.search(regx, sent):
                    res = re.findall(regx, content)[0]
                    if res not in entity_set:
                        entity_set.add(res)
                        modify_entity_list.append(res)
                        flag = False
                        break

        if flag and flag2:
            modify_entity_list.append(ent)

    return modify_entity_list

# 过滤
def filter_entity(entity_list):
    new_ent_list = copy.deepcopy(entity_list)
    for ent in entity_list:
        ent = ent.decode('utf-8')
        remove_flag = True
        for suff in ENT_SUFFIX_LIST:
            if suff in ent:
                remove_flag = False
                re_ent = ent.replace(suff, '')
                if len(re_ent) < 4:
                    new_ent_list.remove(ent)
                    break

        if remove_flag and len(ent)>4:
            new_ent_list.remove(ent)


    return new_ent_list


def recongize(ner_entity_list, ner_content, first_role, second_role):
    entity_role_list = []
    for ent in ner_entity_list:
        flag = True

        # 前缀first_role
        if second_role + ent in ner_content:
            entity_role_list.append((ent, second_role))
            flag = False

        elif first_role + ent in ner_content:
            entity_role_list.append((ent, first_role))
            flag = False


        elif u'第三人' + ent in ner_content:
            flag = False
            continue

        # 诉 定位
        if flag and u'诉' in ner_content:
            subsent_list = re.split(u'诉', ner_content)
            if ent in subsent_list[0]:
                entity_role_list.append((ent, first_role))
                flag = False
            elif ent in subsent_list[1]:
                entity_role_list.append((ent, second_role))
                flag = False
        # 距离
        if flag:
            location = ner_content.find(ent)
            for i in range(location, -1, -1):
                if i - 4 > 0:
                    if first_role in ner_content[i - 4: i]:
                        entity_role_list.append((ent, first_role))
                        flag = False

                        break
                    elif second_role in ner_content[i - 4: i]:
                        entity_role_list.append((ent, second_role))
                        flag = False

                        break
        if flag:
            entity_role_list.append((ent, u'无法识别角色'))

    return entity_role_list


def isExtract(content):
    content = content.decode('utf-8')
    regx = ur'遗失[\u4e00-\u9fa5]+汇票'
    if re.search(regx, content) or u'票据遗失' in content or u'汇票遗失' in content:
        return False
    else:
        return True


def countNotice(content):
    first_role, second_role = check_content(content)

    if not isExtract(content):
        print '******特例不抽取*******'
        return "特例不抽取"
    else:
        rule_content, ner_content = split_content(content)
        truncation_entity_set, truncation_entity_role_list = truncation_by_rule(rule_content, second_role)

        # print '\n----截取实体 角色----'
        # for ent, role in truncation_entity_role_list:
        #     print ent, role


        ner_entity_set = ner_entity(ner_content, truncation_entity_set)
        try:
            bracket_dict = brackets(content)
            ner_entity_list = modify_ent(ner_content, bracket_dict, ner_entity_set)
        except Exception as e:
            pass

        ner_entity_list = modify_ent2nd(content, ner_entity_list)
        ner_entity_list = filter_entity(ner_entity_list)
        ner_entity_role_list = recongize(ner_entity_list, ner_content, first_role, second_role)

        ner_entity_role_list.extend(truncation_entity_role_list)
        return ner_entity_role_list
        # print '\n----NER实体 角色----'
        # for ent, role in ner_entity_role_list:
        #     print ent, role


if __name__ == "__main__":
    # with open('./notice.json', 'r') as file_obj:
    #     data = file_obj.read()
    # data = json.loads(data)
    # i = 1
    # for d in data:
    #     print '\n------------%s：原文----------' % (i)
    #     content = d['content']
    #     print content
    #     if i == 27:
    #         pass
    #
    #     entity_role_list = countNotice(content)
    #     print '\n---------抽取结果----------'
    #     if entity_role_list and isinstance(entity_role_list, list):
    #         for e, r in entity_role_list:
    #             print e, r
    #     i += 1

    # content_list = [
    #     "佘卫明、岳琼：\r\n    本院受理原告中国邮政储蓄银行股份有限公司绥宁县支行诉你们金融借款合同纠纷一案，已审理终结。现依法向你们公告送达（2017）湘0527民初390号民事判决书，限你们自公告之日起60日内来本院领取民事判决书，逾期则视为送达。如不服本判决，可在公告期满后15日内，向本院递交上诉状及副本，上诉于湖南省邵阳市中级人民法院。逾期未上诉本判决即发生法律效力。",
    #     "贵州明圆贸易有限公司、贵州日月圆房屋开发有限公司：本院受理的原告贵阳农村商业银行股份有限公司小河支行诉你们金融借款合同纠纷一案，已审理终结，现依法向你公告送达本院（2013）花民商字第38号民事判决书。限你自公告发布之日起60日内到本院领取民事判决书，逾期视为送达。如不服本判决，可在公告期满后15日内，向本院递交上诉状及副本，上诉于贵州省贵阳市中级人民法院，逾期则本判决发生法律效力。",
    #     "吴宝恩、罗忠翠、刘振山、朱龙凤、林发生、王世萍、吴宗启、赵玉凤、付东亮、吴传忠：本院受理原告中国邮政储蓄银行股份有限公司大连瓦房店市支行与你们金融借款合同纠纷一案，现已审理终结。依法向你们公告送达(2017)辽0281民初1867号民事判决书。限你们自公告之日起60日内来本院第十四审判庭领取民事判决书，逾期则视为送达。如不服本判决，可在公告送达期满之日起l5日内向本院递交上诉状及副本，上诉于辽宁省大连市中级人民法院。",
    #     "广东益和堂制药有限公司因遗失银行承兑汇票一张，向本院申请公示催告。本院决定受理。依照《中华人民共和国民事诉讼法》第二百一十九条规定，现予公告。一、公示催告申请人：广东益和堂制药有限公司二、公示催告的票据：银行承兑汇票，号码为30600051 22367073，金额为50000元，出票人为广西九州通医药有限公司，收款人为九州通医药集团股份有限公司，付款行为广发银行南宁分行营业部，出票日期为2017年7月6日，到期日为2018年1月6日，被背书人为广东益和堂制药有限公司。三、申报权利的期间：自公告之日起至2018年1月31日，利害关系人应向本院申报权利。届时如果无人申报权利，本院将依法作出判决，宣告上述票据无效。在公示催告期间，转让该票据的行为无效。",
    #     "吴宝恩、罗忠翠、刘振山、朱龙凤、林发生、王世萍、吴宗启、赵玉凤、付东亮、吴传忠：本院受理原告中国邮政储蓄银行股份有限公司大连瓦房店市支行与你们金融借款合同纠纷一案，现已审理终结。依法向你们公告送达(2017)辽0281民初1867号民事判决书。限你们自公告之日起60日内来本院第十四审判庭领取民事判决书，逾期则视为送达。如不服本判决，可在公告送达期满之日起l5日内向本院递交上诉状及副本，上诉于辽宁省大连市中级人民法院。",
    #     "广东益和堂制药有限公司因遗失银行承兑汇票一张，向本院申请公示催告。本院决定受理。依照《中华人民共和国民事诉讼法》第二百一十九条规定，现予公告。一、公示催告申请人：广东益和堂制药有限公司二、公示催告的票据：银行承兑汇票，号码为30600051 22367073，金额为50000元，出票人为广西九州通医药有限公司，收款人为九州通医药集团股份有限公司，付款行为广发银行南宁分行营业部，出票日期为2017年7月6日，到期日为2018年1月6日，被背书人为广东益和堂制药有限公司。三、申报权利的期间：自公告之日起至2018年1月31日，利害关系人应向本院申报权利。届时如果无人申报权利，本院将依法作出判决，宣告上述票据无效。在公示催告期间，转让该票据的行为无效。",
    #     "2016年7月27日，本院依法裁定债务人威海市金诺房地产开发有限公司重整，并通过随机方式指定威海铭信清算事务所有限公司担任管理人（通讯地址：威海经济技术开发区蓝星万象城-23号-A805，联系电话：0631-5910988、15554478329，联系人：王晓军）。2017年8月23日，本院依法裁定债务人文登市惠和房地产开发有限公司重整。2017年8月31日，本院根据管理人的申请，认定文登市惠和房地产开发有限公司与威海市金诺房地产开发有限公司主体混同，依法裁定该两公司合并重整，并决定由威海铭信清算事务所有限公司担任两公司合并重整管理人。文登市惠和房地产开发有限公司的债权人应于本公告发布之日起90日内向管理人申报债权，申报地点同上。债权申报须知及申报格式文本，请到管理人网站http：//mxqingsuan.com查阅、下载。文登市惠和房地产开发有限公司的债务人或者财产持有人应向管理人清偿债务或交付财产。第一次债权人会议定于2018年1月9日上午9时在威海经济技术开发区东海路28号(凤林龙凤社区)四楼会议室召开。",
    #     "本院于2017年6月9日立案受理申请人宁波松乐继电器有限公司的公示催告申请，对其遗失的银行承兑汇票(票号31400051／28772703、出票人江苏金能电气科技有限公司、收款人江苏世纪金元机电有限公司、付款行江苏扬中农村商业银行清算中心、出票日期2016年8月16日，汇票到期日2017年2月16日、出票金额10000元)，依法办理了公示催告手续。公示催告期间无人申报权利。本院于2017年9月6日判决：一、宣告申请人宁波松乐继电器有限公司持有的票号31400051／28772703、票面金额10000元的银行承兑汇票无效；二、自本判决公告之日起，申请人宁波松乐继电器有限公司有权向支付人请求支付。",
    #     "申请人凤阳县磊雨石英砂有限公司因银行承兑汇票遗失，向本院申请公示催告。本院决定受理。依照《中华人民共和国民事诉讼法》第二百一十九条规定，现予公告。一、公示催告申请人：凤阳县磊雨石英砂有限公司。二、公示催告的票据：银行承兑汇票一张(票号为30200053 27003057、金额为44265.7元、出票人为陕西兴庆医药有限公司、收款人为河北万岁医药集团有限公司、最后持票人为凤阳县磊雨石英砂有限公司、付款行为中信银行西安分行账务中心、出票日期为2016年12月27日、汇票到期日为2017年6月27日)。三、申报权利的期间：自公告之日起60日内。四、自公告之日起60日内，利害关系人应向本院申报权利。届时如果无人申报权利，本院将依法作出判决，宣告上述票据无效。在公示催告期间，转让该票据权利的行为无效。",
    #     "王思思：本院受理天津滨海新区天保小额贷款有限公司诉天津七建建筑工程有限公司、王国扬、王思思、天津市南开国有资产经营投资有限公司、天津市南开中小企业非融资性信用担保有限公司企业借贷纠纷一案，上诉人天津市南开区国有资产经营投资有限公司就（2016）津02民初393号判决书提起上诉。现依法向你公告送达上诉状副本。自公告之日起，经过60日即视为送达。提出答辩状的期限为公告期满后15日内，逾期将依法审理。",
    #     "侯金超、张美玲：\\r\\n本院受理原告中国光大银行股份有限公司天津梅江支行诉你方借款合同纠纷一案，现依法向你公告送达起诉状副本、诉讼通知书、举证通知书、合议庭组成人员通知书及开庭传票。自公告之日起经过60日即视为送达。提出答辩状的期限和举证的期限分别为公告送达期满后的15日内和30日内。本院定于举证期满后的第三日上午9时（遇法定节假日顺延）在本院第三法庭公开开庭审理，逾期将依法缺席判决。\\r\\n天津市西青区人民法院",
    #     "丁年春、黄山市休宁县锦华房地产开发有限公司：本院在执行黄山永信非融资性担保有限公司与丁年春、黄山市休宁县锦华房地产开发有限公司民间借贷纠纷一案中，依法委托芜湖华瑞房地产土地资产评估工程咨询有限公司对黄山市休宁县锦华房地产开发有限公司所有的位于安徽省休宁县南城御景2幢2-902室房屋进行评估，该房屋市场评估价值为283713元(单价3186元／m2)。现依法公告送达房地产估价报告。自发出公告之日起60日即视为送达。如对评估报告有异议，自公告送达期满后次日起10日内书面向本院提出，逾期不提异议，本院将依法公开拍卖。",
    #     "亿隆宝（北京）投资有限责任公司、中融诚咨询服务股份有限公司、唐瑞莺、李钰、唐金同、董尾妹：本院受理原告杭州银行股份有限公司北京分行诉你们金融借款合同纠纷一案，现依法向你们公告送达起诉状副本及开庭传票。自公告之日起经过60日，即视为送达。提出答辩状和举证的期限为公告期满后15日和30日内。并定于举证期满后第3日下午14时（遇法定假日顺延）在本院第13法庭开庭审理，逾期将依法缺席裁判。",
    #     "饶新锦、张著安：本院受理原告赣州开发区潭东镇振兴钢管租赁部诉被告饶新锦、张著安、福建省惠东建筑工程有限公司租赁合同纠纷一案，上诉人福建省惠东建筑工程有限公司就(2015)赣开民二初字第1426号判决书提起上诉。现依法向你公告送达上诉状副本。自公告之日起，经过60日即视为送达。提出答辩状的期限为公告期满后15日内，逾期将依法审理。",
    #     "天津东方鹏盛商贸有限公司,天津鼎泰恒瑞投资集团有限公司,天津市众鹏物资有限公司,天津市厚德投资发展有限公司,范德秀,朱智星,高志华：高志红就(2016)津01民初519号民事判决书提起上诉，本院已受理。现依法向你公告送达上诉状副本。自公告之日起经过60日即视为送达。提出答辩状的期限为公告期满后15日内。逾期本院将依法作出处理。",
    #     "北京鼎丰源投资有限公司、中融诚咨询服务股份有限公司、唐金耀、唐淑如：本院受理原告杭州银行股份有限公司北京分行诉你们金融借款合同纠纷一案，现依法向你们公告送达起诉状副本及开庭传票。自公告之日起经过60日，即视为送达。提出答辩状和举证的期限为公告期满后15日和30日内。并定于举证期满后第3日下午14时（遇法定假日顺延）在本院第13法庭开庭审理，逾期将依法缺席裁判。",
    #     "沈阳吉顺轩（北京）家具有限公司：本院受理原告宋晓伟、戴悦诉你方及沈阳吉顺轩家具有限公司产品销售者责任纠纷一案，已审理终结。因原告深圳市香江商业管理有限公司沈阳分公司不服本院作出的（2017）辽0106民初336号民事判决在法定期限内提出上诉，现依法向你公告送达上诉状。限你自公告之日起60日内来本院第二十一法庭领取上诉状，逾期则视为送达。",
    #     "沈阳吉顺轩(北京)家具有限公司：本院受理原告宋晓伟、戴悦诉你方及沈阳吉顺轩家具有限公司产品销售者责任纠纷一案，已审理终结。因原告深圳市香江商业管理有限公司沈阳分公司不服本院作出的（2017）辽0106民初336号民事判决在法定期限内提出上诉，现依法向你公告送达上诉状。限你自公告之日起60日内来本院第二十一法庭领取上诉状，逾期则视为送达。",
    #
    # ]

    content_list = [
        '襄阳市胜男实业有限公司、襄阳广荣实业有限公司、李群丽、刘萍、詹伟明：原告汉口银行股份有限公司襄阳分行与被告襄阳市胜男实业有限公司、襄阳大汉光武酿酒股份有限公司、襄阳广荣实业有限公司、刘荣军、李群丽、刘萍、詹伟明金融借款合同纠纷一案已审理终结。现依法向你们送达（2017）鄂06民初87号民事判决书，自公告之日起60日内来本院领取民事判决书，期满则视为送达。如不服本判决，可在公告期满15日内，向本院递交上诉状及副本，上诉于湖北省高级人民法院。逾期本判决即发生法律效力。',
        '本院根据申请人中国邮政储蓄银行股份有限公司泰州市分行的申请于2017年9月18日裁定受理被申请人泰州昌升模具制造有限公司（以下简称昌升公司）破产清算一案，并于2017年9月18日指定兴化兴财会计事务所有限公司担任昌升公司破产管理人。昌升公司的债权人应自公告之日起30日内，向昌升公司破产管理人（通信地址：泰州市海陵区苏陈镇夏棋村昌升公司厂内；邮政编码：225300；联系人：范先生，联系电话:13961068600）申报债权。未在上述期限内申报债权的，可以在破产财产分配方案提交债权人会议讨论前补充申报，但对此前已进行的分配无权要求补充分配，同时要承担为审查和确认补充申报债权所产生的费用。未申报债权的，不得依照《中华人民共和国企业破产法》规定的程序行使权利。昌升公司的债务人或者财产持有人应当向昌升公司破产管理人清偿债务或交付财产。本院定于2017年11月29日上午9时30分在泰州市海陵区人民法院第505法庭（地址：江苏省泰州市海陵区春晖路98号）召开第一次债权人会议。依法申报债权的债权人有权参加债权人会议。参加会议的债权人系法人或其他组织的，应提交营业执照、法定代表人或负责人身份证明书;参加会议的债权人系自然人的，应提交个人身份证明。如委托代理人出席会议，应提交特别授权委托书、委托代理人的身份证件或律师执业证，委托代理人是律师的还应提交律师事务所的指派函。',
        '深圳市前海华燃能源有限公司、广西九基股权投资基金管理有限公司、东莞市旭源石油化工有限公司、山东港燃能源有限公司、黄君庆、马建江、邱洁泉、伍国莉：本院受理原告陈家枫诉你们借款合同纠纷一案，现依法向你们公告送达(2017)桂0103民初第6179号案起诉状副本及证据、开庭传票等应诉材料。自发出公告之日起经过60日，即视为送达。提出答辩状的期限和举证期限分别为公告期满后15日和30日内。并定于举证期满后第3日上午8时30分(遇法定节假日顺延)在本院三楼8号法庭开庭审理，逾期将依法缺席裁判。',
        '赵小军、薛会娥、赵四、胡银芬：本院受理伊金霍洛金谷村镇银行股份有限公司诉赵小军、薛会娥、赵四、胡银芬金融借款合同纠纷一案，已审理终结。现依法向你公告送达（2017）内0627民初1800号民事判决书。自公告之日起，60日内来本院领取民事判决书，逾期则视为送达。如不服本判决，可在公告期满后15日内，向本院递交上诉状及副本，上诉于内蒙古鄂尔多斯市中级人民法院。逾期本判决即发生法律效力。',
        '缪恩来：本院受理原告田利东与被告缪恩来、怀宁县江镇代家凹铜矿有限公司、曹徽民间借贷纠纷一案，现依法向你公告送达起诉状副本、应诉通知书、举证通知书、（2016）冀04民初64号民事裁定书、（2016）冀04民初64-1民事裁定书等诉讼文书，自公告之日起满三个月内来本院领取，逾期视为送达。',
        '石家庄英杰建筑装饰工程有限公司：本院受理原告孟庆明与被告石家庄一建建设集团有限公司、第三人沧州市天成房地产开发有限公司建设工程施工合同纠纷一案，被告石家庄一建建设集团有限公司申请追加你公司为本案被告，现依法向你公司公告送达参加诉讼通知书、起诉状副本、应诉通知书、举证通知书、诉讼风险提示书及开庭传票。自公告之日起，经过60日即视为送达。提出答辩状的期限和举证期限为公告期满后15日内。并定于举证期满后第3日上午9时（遇法定假日顺延）在本院民事第二审判庭开庭，逾期将依法缺席判决。',
        '刘卫东、中晟路桥建筑有限公司西安分公司：我院受理原告刘世华与被告刘卫东、中晟路桥建筑有限公司西安分公司房屋租赁合同纠纷一案，现依法向你公告送达起诉状副本，起诉要点：1、判决解除双方租赁合同，立即向原告交还租赁房屋；2、判决被告向原告支付拖欠租金156223.13元，违约金68273*2%*11个月=15020.06；合计171243.19元；3、判令被告承担本案的全部诉讼费用。自本公告发出之日起，经过六十日，即视为送达。提出答辩状和举证的期限为公告送达期满后的15日内。并定于举证期满后的第一日上午十时四十五分（遇节假日顺延）在本院第六审判庭开庭审理，逾期将依法缺席判决。',
        '刘付生：本院受理原告鹤壁市大八角建筑设备租赁站与被告刘付生、鹤壁市华夏建筑安装有限公司租赁合同纠纷一案。现依法向你公告送达起诉状副本、应诉通知书、举证通知书、诉讼风险提示书、廉政监督卡、开庭传票、告知审判庭组成人员通知书，自公告发出之日起经过60日即视为送达。提出答辩状和举证的期限均为公告期满后的15日内，并定于举证期满后的第3日上午9时00分（遇法定节假日顺延）在本院第七审判庭公开开庭审理，逾期将依法缺席裁判。',
        '王可政、连丽：本院在执行东亚银行（中国）有限公司大连分行申请执行与你金融借款合同纠纷一案中，申请执行人东亚银行（中国）有限公司大连分行于2017年8月22日向本院申请执行。现因你下落不明，依法向你公告送达本院(2017)辽0202执2428号失信决定书和执行通知书，该失信决定书自作出之日起生效；该执行通知书自公告登报之日起六十日内视为送达，并于公告期满后五日内来院履行大连市中山区人民法院(2016)辽0202民初字第2852号民事判决书所确定的义务，逾期不予履行，本院将对被执行人王可政名下位于大连市经济技术开发区现代城3栋-6-14号房屋进行评估。定于被执行人履行义务期满后的第一周的星期二上午九时在大连市中级人民法院审判法庭三楼二室进行摇号选定评估机构，逾期不再办理。',
        '张久高、陈洪杰、赵会彬、郭玉莲、徐友、刘广芝、刘玉柱、张传丽、王枝有、孙丽晶：本院受理原告中国邮政储蓄银行股份有限公司勃利县支行诉你们金融借款合同纠纷一案，现依法向你们公告送达起诉状副本、应诉通知书、举证通知书及开庭传票。自公告发出之日起60日内即视为送达。提出答辩状的期限和举证期限分别为公告送达期满后的15日和30日内。并定于举证期满后的第三个工作日上午9时在勃利县人民法院青山法庭公开开庭审理，逾期将依法缺席判决。',
        '湖南大成粮油购销有限公司、曾向东、蔡旭华、周共武：原告贺路亭诉你们及第三人李春光、谢叶波民间借贷纠纷一案，本院已审理终结。现依法向你们公告送达（2015）娄星民一初字第2499号民事判决书。自公告之日起60日内来本院领取民事判决书，逾期即视为送达。如不服本判决，可在公告期满后15日内，向本院递交上诉状及副本，上诉于娄底市中级人民法院。逾期本判决即发生法律效力。',
        '姚杰、杨金艳、杨丽丹、曹茂云、陈江溢、高清莲：本院受理原告湖南省中小企业信用担保有限责任公司诉被告鼎鑫元（湖南）化工机械制造有限公司、姚杰、杨金艳、杨丽丹、曹茂云、陈江溢、高清莲追偿权纠纷一案，现已审理终结。现依法向你们公告送达本院(2016)湘0111民初9463号民事判决书。自公告之日起60日内来本院领取民事判决书，逾期视为送达。如不服本判决，可以在公告期满后15日内，向本院递交上诉状及副本，上诉于湖南省长沙市中级人民法院。逾期本判决即发生法律效力。',
        '黄通腾、张婧宇：本院受理原告林久跳与你二人及北京市港龙腾达智能卡科技有限公司承揽合同纠纷一案，因无法向你二人送达开庭传票、起诉书副本、证据等材料，现依法向你们公告送达起诉状副本、应诉通知书、举证通知书、开庭传票。林久跳起诉请求判令三被告连带双倍返还原告定金30000元；诉讼费由被告承担。自本公告发出之日起60日即视为送达。提出答辩状和举证的期限均为公告期满后的15日内。并定于举证期届满后第三日上午九时（遇法定节假日顺延）在本院第十法庭公开开庭审理此案，逾期本院将依法缺席裁判。',
        '王君：本院受理原告中国农业银行股份有限公司锦州市府路支行诉你信用卡纠纷一案，案号（2017）辽0792民初734号，现依法向你公告送达起诉状副本、应诉及举证通知书、诉讼风险告知书、开庭传票。自公告之日起经过60日，即视为送达。提出答辩状和举证期限分别为公告期满后的15日内和30日内。并定于举证期满后第3日上午9时（遇法定假日顺延）在锦州市松山新区人民法庭一号法庭开庭审理，逾期将依法缺席裁判。',
        '北京仁和信商贸有限公司：本院受理原告余建新诉被告北京市工商行政管理局西城分局工商登记一案（案号为：2017京0102行初512号），因你公司与本案被诉行政行为具有法律上的利害关系，本院依法通知你公司作为本案的第三人参加诉讼。现依法向你公司送达参加诉讼通知书、起诉状副本及原告证据材料、诉讼须知、廉政监督卡、答辩状及被告证据材料、开庭传票、合议庭组成人员及书记员告知书。自公告之日起经过60日即视为送达。提出答辩状的期限为公告送达期满后的15日内。法院指定的举证期限为公告送达期满后30日内。并定于举证期届满后的第3日上午9点（遇节假日顺延）在本院第五法庭（第二审判区）开庭审理。逾期未到庭参加诉讼的，不影响本案的审理，本院将依法审判',
        '陈小飘：原告北京中和东方资产管理有限公司诉被告陈小飘借款合同纠纷一案，现依法向你们公告送达起诉状副本、应诉通知书及开庭传票。自本公告发出之日起经过60日即视为送达。提出答辩状和举证的期限分别为公告送达期满后的15日和15日内，并定于举证期满后的第3日上午9时00分（遇节假日顺延）在本院第五号法庭公开开庭审理，逾期将依法缺席审判。',
        '关运科：本院受理原告綦伟诉你民间借贷纠纷一案，已审理终结。因其他方式无法送达，现依法向你公告送达本院(2017)辽0203民初344号民事判决书。判决被告大连信开数码有限公司给付原告綦伟借款1500000元及利息300000元、律师费22000，合计1822000元。被告大连信开投资公司、大连富士山办公设备有限公司、大连永佐环保制品有限公司、关运科承担连带给付责任。限你自公告之日起60日内来本院领取判决书，逾期则视为送达。如不服本判决，可在公告期满之日起15日内向本院递交上诉状及副本，上诉于辽宁省大连市中级人民法院。逾期本判决即发生法律效力。',
        '李晓慧：本院受理原告大庆昊方房地产开发有限公司诉你及第三人兴业银行股份有限公司大庆分行房屋买卖合同纠纷一案，现依法向你送达起诉状副本、应诉通知书、举证通知书、合议庭组成人员通知书、传票。自公告之日起，60日即视为送达。提出答辩状和举证期限分别为公告送达期满后15日和30日内。开庭日期定于举证期满后第3日上午8时40分（遇法定假日顺延）在本院青龙山法庭公开开庭审理，逾期将依法缺席判决。',
        '李纪蓉：本院执行的招商银行股份有限公司成都分行申请执行你金融借款合同纠纷一案，本院依法对你所有的位于成都市锦江区东大街芷泉段88号6栋1单元33层3301号房屋予以委托评估，评估价为4020000元。因你下落不明，现依法向你送告送达评估报告及( 2017)川0107执591号执行裁定书。经过60日，即视为送达。如你从送达之日起10日内未向本院提交书面异议又未履行生效法律文书确定的义务，本院将依法对上述财产予以拍卖。',
        '要铭、何邦安、何哲、明维、石嘴山市要铭家具陶瓷有限公司、宁夏尚果装饰设计工程有限公司：本院受理的原告宁夏石嘴山农村商业银行股份有限公司与被告要铭、何邦安、何哲、明维、任小平、王柳郁、石嘴山市要铭家具陶瓷有限公司、宁夏尚果装饰设计工程有限公司金融借款合同纠纷一案，现依法向你们公告送达起诉状副本、证据副本、举证通知书、应诉通知书、转换程序通知书、（2017）宁0202民初1747号民事裁定书及开庭传票。本案原告诉讼请求：1.依法判令被告要铭偿还原告宁夏石嘴山农村商业银行股份有限公司的借款本金200万元，利息845916.42元(利息截止2017年5月20日)，共计2845916.42元，并按合同约定的月利率支付从2017年5月21日起至贷款本息清偿完毕期间的利息；2.依法判令被告何哲、明维、任小平、王柳郁、何邦安、石嘴山市要铭家具陶瓷有限公司、宁夏尚果装饰设计工程有限公司对第一项诉讼请求的款项承担连带清偿责任；3.本案诉讼费、公告费、执行费由被告承担。本公告发出之日起60日即视为送达，提出答辩状的期限为公告期满后的15日内，举证期为公告送达期满后30日内，并定于举证期满后的第3日下午15时（遇节假日顺延）在本院第九审判庭公开开庭审理此案，逾期将依法缺席判决。',
        '北京市巨龙工程有限公司：上诉人太原市晋源区龙逊达管件经销部就(2017）晋0109民初1120号民事判决书提起上诉。现依法向你公司公告送达上诉状副本(主要内容为：依法撤销(2017）晋0109民初1120号民事判决书，改判被上诉人北京市巨龙工程有限公司、北京市巨龙工程有限公司山西分公司共同承担给付上诉人货款的责任；改判支持上诉人一审的利息损失按月息2%赔偿上诉人从2014年10月21日至货款全部还清之日止的利息损失。）自本公告发出之日起经过60日即视为送达。提出答辩状的期限为公告期满后15日内，逾期将依法审理。',
        '上海军八集装箱配件厂、曹友良(310225195510173053)：本院于2017年8月31日立案受理了原告海盐中达金属电子材料有限公司诉被告上海军八集装箱配件厂、曹友良买卖合同纠纷一案，因你下落不明，故现依法向你公告送达起诉状副本、应诉通知书、举证通知书和开庭传票等材料。原告请求法院判决被告支付原告货款153386.69元等。自发出公告之日起经过60日，即视为送达。提出答辩状的期限和举证期限分别为公告期满后的15日和30日内。开庭时间定于举证期限届满后第3日(如遇节假日顺延)的上午9时00分在本院沈荡人民法庭第三法庭依法组成合议庭公开开庭审理，逾期将依法缺席裁判。',
        '上海市普陀区威凯盛南北干货行、上海百和投资管理有限公司、王蓉、赵飞飞：本院受理原告杭州银行股份有限公司上海分行诉你们及上海市普陀区昌海盛南北干货行金融借款合同纠纷一案，已审理终结。现依法向你们公告送达（2017）沪0101民初2717号民事判决书。自公告之日起，60日内来本院民五庭领取民事判决书，逾期则视为送达。如不服本判决，可在判决书送达之日起15日内，向本院递交上诉状，并按对方当事人的人数提出副本，上诉于上海市第二中级人民法院。逾期本判决即发生法律效力',
        '陈建平、陈慧兰：本院受理的湖南省小微企业金融服务促进会诉被告陈建平、陈慧兰、第三人中国民生银行股份有限公司长沙分行追偿权纠纷一案，现依法向你们公告送达（2016）湘0104民初7946号民事判决书，限你自公告之日起60日内来本院领取判决书，逾期则视为送达。如不服本判决，可自本判决书送达之日起15日内向本院递交上诉状，并按对方当事人的人数提出副本，上诉于湖南省长沙市中级人民法院。逾期则本判决发生法律效力。',
        '蓬江区铭恒五金制品厂（经营者：许乃专）：本院受理原告江门市泓林装饰材料有限公司诉你买卖合同纠纷一案，因无法联系你，现依法向你公告送达起诉书副本、应诉通知书、举证通知书、合议庭组成人员通知书及开庭传票。原告起诉要求：1、被告支付货款457000元及逾期付款利息(从起诉之日起，按照中国人民银行同期同类逾期贷款利率计算至被告付清之日止)给原告；2、本案的诉讼费由被告承担。自本公告发出之日起，经过60日即视为送达。提出答辩和举证的期限均为公告送达期满后的15日内，并定于举证期满后的第3日下午15时(遇法定假顺延)在本院荷塘法庭开庭审理。逾期将依法缺席判决。',
        '葛昀：本院在执行招商银行股份有限公司大连分行申请执行与你金融借款合同纠纷一案中，申请执行人招商银行股份有限公司大连分行于2017年5月15日向本院申请执行。现因你下落不明，依法向你公告送达本院(2017)辽0202执1644号失信决定书和执行通知书，该失信决定书自作出之日起生效；该执行通知书自公告登报之日起六十日内视为送达，并于公告期满后五日内来院履行大连市中山区人民法院(2016)辽0202民初字第2810号判决书所确定的义务，逾期不予履行，本院将对被执行人葛昀所有的位于大连经济技术开发区东阁里46栋7-2-1号房产进行评估。定于被执行人履行义务期满后的第一周的星期二上午九时在大连市中级人民法院审判法庭三楼二室进行摇号选定评估机构，逾期不再办理。',
        '陈英达、徐青梅、中融诚咨询服务股份有限公司：本院受理原告杭州银行股份有限公司北京安贞支行诉你们信用卡纠纷一案，现依法向你们公告送达民事起诉状副本、应诉通知书、举证通知书及开庭传票。自公告之日起60日视为送达。提出答辩状的时限为送达期满后15日。本院将于答辩期满后第1个工作日上午9时（如遇法定假日顺延）在北京市朝阳区人民法院金融审判庭（北京市朝阳区小红门乡龙爪树村甲10号）开庭审理，逾期将依法缺席判决。',
        '河南京安商务发展股份有限公司：原告上海奇显建筑设计咨询有限公司诉被告河南京安商务发展股份有限公司建设工程设计合同纠纷一案，现已审理终结。因你们下落不明，依照《中华人民共和国民事诉讼法》第九十二条的规定，向你们公告送达本院（2017）沪0101民初5560号民事判决本院判决如下：被告河南京安商务发展股份有限公司应于本判决生效之日起十日内向原告上海奇显建筑设计咨询有限公司支付设计咨询费92,500元。如不服本判决，可在判决书送达之日起十五日内，向本院递交上诉状，并按对方当事人的人数提出副本，上诉于上海市第二中级人民法院。特此公告。上海市黄浦区人民法院二〇一七年八月二十八日',
        '王福力、王通、三井住友融资租赁（中国）有限公司上海分公司:本院对原告北京圣凯华瑞工程机械有限公司与被告王福力、王通、第三人三井住友融资租赁（中国）有限公司上海分公司融资租赁合同纠纷一案，现依法向你公告送达起诉状副本、证据、应诉通知书及开庭传票。自公告发出之日起60日视为送达。提出答辩状的时限为送达期满后的15日。本院将于答辩期满后第1个工作日上午9时（如遇法定节假日顺延）在北京市朝阳区人民法院金融审判庭（北京市朝阳区小红门乡龙爪树村甲10号）开庭审理，逾期将依法缺席判决。',
        '黑龙江省沷雪泉酒业有限公司：本院于2017年8月3日刊登在人民法院报G29版的九三粮油工业集团有限公司诉被告牡丹江市东安区中意粮油经销处、你方买卖合同纠纷一案的公告中，其中“举证期满后第3日”“应更正为公告期满后第3日”特此更正。',
        '张桂、姜春灵、襄阳市万里路汽车销售服务有限公司：本院依法受理原告中国农业银行股份有限公司襄阳樊城支行诉你们金融借款合同纠纷一案，现依法向你们公告送达起诉状副本、应诉通知书、举证通知书及开庭传票。自发出公告之日起，经过60日即视为送达。提出答辩状和举证的期限分别为公告送达期满后的15日和30日内，并定于举证期满后第3日上午10时（遇法定节假日顺延）在本院第三审判庭开庭进行审理，逾期将依法缺席裁判。',
        '刘乐：本院受理原告招商银行股份有限公司北京分行诉你金融借款合同纠纷一案，原告要求本院判令你偿还借款本金、罚息，并承担原告律师费损失及本案全部诉讼费用。现依法向你公告送达起诉状副本、应诉通知书、举证通知书及开庭传票、证据、合议庭组成人员通知书。自公告之日起经过60日，即视为送达。提出答辩状的期限和举证期限分别为公告期满后15日和30日内，并定于举证期满后次日下午14时（遇法定节假日顺延）在本院金融街法庭（北京市西城区广成街4号院2号楼）第七法庭公开开庭审理此案，逾期将依法缺席裁判。',
        '高绪宪：本院受理原告吉林市春城加油站有限公司诉被告曲志友、高绪宪合同纠纷一案已审理终结。现依法向你公告送达（2017）吉0104民初2890号民事判决书。自公告之日起60日内来本院领取民事判决书，逾期则视为送达。如不服本判决，可在公告期满后15日内，向本院递交上诉状及副本，上诉于吉林省长春市中级人民法院。逾期本判决即发生法律效力。',
        '杨静茹：我院已受理原告杨贵美诉你、王永伟、王耀方、中国人民财产保险股份有限公司孟津支公司机动车交通事故赔偿责任纠纷一案。因你下落不明，现依法向你公告送达起诉状副本、应诉通知书、举证通知书、开庭传票及廉政监督卡等应诉法律文书。自本公告发出之日起，经过六十日即视为送达。提出答辩状及举证的期限分别为公告送达期满后的十五日和三十日内。并定于举证期满后第三日上午9时00分(遇法定节假日顺延)在本院第二审判庭公开开庭审理此案，逾期本院将依法缺席判决。',
        "佘卫明、岳琼：\r\n    本院受理原告中国邮政储蓄银行股份有限公司绥宁县支行诉你们金融借款合同纠纷一案，已审理终结。现依法向你们公告送达（2017）湘0527民初390号民事判决书，限你们自公告之日起60日内来本院领取民事判决书，逾期则视为送达。如不服本判决，可在公告期满后15日内，向本院递交上诉状及副本，上诉于湖南省邵阳市中级人民法院。逾期未上诉本判决即发生法律效力。",
        "贵州明圆贸易有限公司、贵州日月圆房屋开发有限公司：本院受理的原告贵阳农村商业银行股份有限公司小河支行诉你们金融借款合同纠纷一案，已审理终结，现依法向你公告送达本院（2013）花民商字第38号民事判决书。限你自公告发布之日起60日内到本院领取民事判决书，逾期视为送达。如不服本判决，可在公告期满后15日内，向本院递交上诉状及副本，上诉于贵州省贵阳市中级人民法院，逾期则本判决发生法律效力。",
        "吴宝恩、罗忠翠、刘振山、朱龙凤、林发生、王世萍、吴宗启、赵玉凤、付东亮、吴传忠：本院受理原告中国邮政储蓄银行股份有限公司大连瓦房店市支行与你们金融借款合同纠纷一案，现已审理终结。依法向你们公告送达(2017)辽0281民初1867号民事判决书。限你们自公告之日起60日内来本院第十四审判庭领取民事判决书，逾期则视为送达。如不服本判决，可在公告送达期满之日起l5日内向本院递交上诉状及副本，上诉于辽宁省大连市中级人民法院。",
        "广东益和堂制药有限公司因遗失银行承兑汇票一张，向本院申请公示催告。本院决定受理。依照《中华人民共和国民事诉讼法》第二百一十九条规定，现予公告。一、公示催告申请人：广东益和堂制药有限公司二、公示催告的票据：银行承兑汇票，号码为30600051 22367073，金额为50000元，出票人为广西九州通医药有限公司，收款人为九州通医药集团股份有限公司，付款行为广发银行南宁分行营业部，出票日期为2017年7月6日，到期日为2018年1月6日，被背书人为广东益和堂制药有限公司。三、申报权利的期间：自公告之日起至2018年1月31日，利害关系人应向本院申报权利。届时如果无人申报权利，本院将依法作出判决，宣告上述票据无效。在公示催告期间，转让该票据的行为无效。",
        "吴宝恩、罗忠翠、刘振山、朱龙凤、林发生、王世萍、吴宗启、赵玉凤、付东亮、吴传忠：本院受理原告中国邮政储蓄银行股份有限公司大连瓦房店市支行与你们金融借款合同纠纷一案，现已审理终结。依法向你们公告送达(2017)辽0281民初1867号民事判决书。限你们自公告之日起60日内来本院第十四审判庭领取民事判决书，逾期则视为送达。如不服本判决，可在公告送达期满之日起l5日内向本院递交上诉状及副本，上诉于辽宁省大连市中级人民法院。",
        "广东益和堂制药有限公司因遗失银行承兑汇票一张，向本院申请公示催告。本院决定受理。依照《中华人民共和国民事诉讼法》第二百一十九条规定，现予公告。一、公示催告申请人：广东益和堂制药有限公司二、公示催告的票据：银行承兑汇票，号码为30600051 22367073，金额为50000元，出票人为广西九州通医药有限公司，收款人为九州通医药集团股份有限公司，付款行为广发银行南宁分行营业部，出票日期为2017年7月6日，到期日为2018年1月6日，被背书人为广东益和堂制药有限公司。三、申报权利的期间：自公告之日起至2018年1月31日，利害关系人应向本院申报权利。届时如果无人申报权利，本院将依法作出判决，宣告上述票据无效。在公示催告期间，转让该票据的行为无效。",
        "2016年7月27日，本院依法裁定债务人威海市金诺房地产开发有限公司重整，并通过随机方式指定威海铭信清算事务所有限公司担任管理人（通讯地址：威海经济技术开发区蓝星万象城-23号-A805，联系电话：0631-5910988、15554478329，联系人：王晓军）。2017年8月23日，本院依法裁定债务人文登市惠和房地产开发有限公司重整。2017年8月31日，本院根据管理人的申请，认定文登市惠和房地产开发有限公司与威海市金诺房地产开发有限公司主体混同，依法裁定该两公司合并重整，并决定由威海铭信清算事务所有限公司担任两公司合并重整管理人。文登市惠和房地产开发有限公司的债权人应于本公告发布之日起90日内向管理人申报债权，申报地点同上。债权申报须知及申报格式文本，请到管理人网站http：//mxqingsuan.com查阅、下载。文登市惠和房地产开发有限公司的债务人或者财产持有人应向管理人清偿债务或交付财产。第一次债权人会议定于2018年1月9日上午9时在威海经济技术开发区东海路28号(凤林龙凤社区)四楼会议室召开。",
        "本院于2017年6月9日立案受理申请人宁波松乐继电器有限公司的公示催告申请，对其遗失的银行承兑汇票(票号31400051／28772703、出票人江苏金能电气科技有限公司、收款人江苏世纪金元机电有限公司、付款行江苏扬中农村商业银行清算中心、出票日期2016年8月16日，汇票到期日2017年2月16日、出票金额10000元)，依法办理了公示催告手续。公示催告期间无人申报权利。本院于2017年9月6日判决：一、宣告申请人宁波松乐继电器有限公司持有的票号31400051／28772703、票面金额10000元的银行承兑汇票无效；二、自本判决公告之日起，申请人宁波松乐继电器有限公司有权向支付人请求支付。",
        "申请人凤阳县磊雨石英砂有限公司因银行承兑汇票遗失，向本院申请公示催告。本院决定受理。依照《中华人民共和国民事诉讼法》第二百一十九条规定，现予公告。一、公示催告申请人：凤阳县磊雨石英砂有限公司。二、公示催告的票据：银行承兑汇票一张(票号为30200053 27003057、金额为44265.7元、出票人为陕西兴庆医药有限公司、收款人为河北万岁医药集团有限公司、最后持票人为凤阳县磊雨石英砂有限公司、付款行为中信银行西安分行账务中心、出票日期为2016年12月27日、汇票到期日为2017年6月27日)。三、申报权利的期间：自公告之日起60日内。四、自公告之日起60日内，利害关系人应向本院申报权利。届时如果无人申报权利，本院将依法作出判决，宣告上述票据无效。在公示催告期间，转让该票据权利的行为无效。",
        "王思思：本院受理天津滨海新区天保小额贷款有限公司诉天津七建建筑工程有限公司、王国扬、王思思、天津市南开国有资产经营投资有限公司、天津市南开中小企业非融资性信用担保有限公司企业借贷纠纷一案，上诉人天津市南开区国有资产经营投资有限公司就（2016）津02民初393号判决书提起上诉。现依法向你公告送达上诉状副本。自公告之日起，经过60日即视为送达。提出答辩状的期限为公告期满后15日内，逾期将依法审理。",
        "侯金超、张美玲：\\r\\n本院受理原告中国光大银行股份有限公司天津梅江支行诉你方借款合同纠纷一案，现依法向你公告送达起诉状副本、诉讼通知书、举证通知书、合议庭组成人员通知书及开庭传票。自公告之日起经过60日即视为送达。提出答辩状的期限和举证的期限分别为公告送达期满后的15日内和30日内。本院定于举证期满后的第三日上午9时（遇法定节假日顺延）在本院第三法庭公开开庭审理，逾期将依法缺席判决。\\r\\n天津市西青区人民法院",
        "丁年春、黄山市休宁县锦华房地产开发有限公司：本院在执行黄山永信非融资性担保有限公司与丁年春、黄山市休宁县锦华房地产开发有限公司民间借贷纠纷一案中，依法委托芜湖华瑞房地产土地资产评估工程咨询有限公司对黄山市休宁县锦华房地产开发有限公司所有的位于安徽省休宁县南城御景2幢2-902室房屋进行评估，该房屋市场评估价值为283713元(单价3186元／m2)。现依法公告送达房地产估价报告。自发出公告之日起60日即视为送达。如对评估报告有异议，自公告送达期满后次日起10日内书面向本院提出，逾期不提异议，本院将依法公开拍卖。",
        "亿隆宝（北京）投资有限责任公司、中融诚咨询服务股份有限公司、唐瑞莺、李钰、唐金同、董尾妹：本院受理原告杭州银行股份有限公司北京分行诉你们金融借款合同纠纷一案，现依法向你们公告送达起诉状副本及开庭传票。自公告之日起经过60日，即视为送达。提出答辩状和举证的期限为公告期满后15日和30日内。并定于举证期满后第3日下午14时（遇法定假日顺延）在本院第13法庭开庭审理，逾期将依法缺席裁判。",
        "饶新锦、张著安：本院受理原告赣州开发区潭东镇振兴钢管租赁部诉被告饶新锦、张著安、福建省惠东建筑工程有限公司租赁合同纠纷一案，上诉人福建省惠东建筑工程有限公司就(2015)赣开民二初字第1426号判决书提起上诉。现依法向你公告送达上诉状副本。自公告之日起，经过60日即视为送达。提出答辩状的期限为公告期满后15日内，逾期将依法审理。",
        "天津东方鹏盛商贸有限公司,天津鼎泰恒瑞投资集团有限公司,天津市众鹏物资有限公司,天津市厚德投资发展有限公司,范德秀,朱智星,高志华：高志红就(2016)津01民初519号民事判决书提起上诉，本院已受理。现依法向你公告送达上诉状副本。自公告之日起经过60日即视为送达。提出答辩状的期限为公告期满后15日内。逾期本院将依法作出处理。",
        "北京鼎丰源投资有限公司、中融诚咨询服务股份有限公司、唐金耀、唐淑如：本院受理原告杭州银行股份有限公司北京分行诉你们金融借款合同纠纷一案，现依法向你们公告送达起诉状副本及开庭传票。自公告之日起经过60日，即视为送达。提出答辩状和举证的期限为公告期满后15日和30日内。并定于举证期满后第3日下午14时（遇法定假日顺延）在本院第13法庭开庭审理，逾期将依法缺席裁判。",
        "沈阳吉顺轩（北京）家具有限公司：本院受理原告宋晓伟、戴悦诉你方及沈阳吉顺轩家具有限公司产品销售者责任纠纷一案，已审理终结。因原告深圳市香江商业管理有限公司沈阳分公司不服本院作出的（2017）辽0106民初336号民事判决在法定期限内提出上诉，现依法向你公告送达上诉状。限你自公告之日起60日内来本院第二十一法庭领取上诉状，逾期则视为送达。",
        "沈阳吉顺轩(北京)家具有限公司：本院受理原告宋晓伟、戴悦诉你方及沈阳吉顺轩家具有限公司产品销售者责任纠纷一案，已审理终结。因原告深圳市香江商业管理有限公司沈阳分公司不服本院作出的（2017）辽0106民初336号民事判决在法定期限内提出上诉，现依法向你公告送达上诉状。限你自公告之日起60日内来本院第二十一法庭领取上诉状，逾期则视为送达。",

    ]

    i = 0
    for content in content_list:
        content = content.decode('utf-8')
        if i == 1:
            pass
        # print '\n---------%s 原文----------' % (i)
        # print content
        entity_role_list = countNotice(content)
        # print '\n---------抽取结果----------'
        if entity_role_list and isinstance(entity_role_list, list):
            for e, r in entity_role_list:
                pass
                # print e, r

        i += 1
