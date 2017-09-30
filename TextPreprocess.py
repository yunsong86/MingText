#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: TextPreprocess.py
@time: 17-9-28 下午10:46
@desc: python3.6
"""
import os
import jieba
import pickle
from Log import *
from com_resources import *
from datetime import datetime
from chardet.universaldetector import UniversalDetector
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer


##############################################################
# 分类语料预处理的类
# 语料目录结构：
# corpus
#   |-catergory_A
#     |-01.txt
#     |-02.txt
#   |-catergory_B
#   |-catergory_C
#   ...
##############################################################

# 文本预处理类
class TextPreprocess:
    # 定义词袋对象:data_set
    # Bunch类提供一种key,value的对象形式
    # target_name:所有分类集名称列表
    # label:每个文件的分类标签列表
    # filenames:文件名称
    # contents:文件内容
    dataset_Bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    tfidf_wordbag_Bunch = Bunch(target_name=[], label=[], filenames=[], tfidf=[], vocabulary={})

    def __int__(self):  # 构造方法
        self.corpus_path = ""  # 原始语料路径
        self.pos_path = ""  # 预处理后语料路径
        self.segment_path = ""  # 分词后语料路径
        self.model_path = ""  # 词袋模型路径
        self.stopword_path = ""  # 停止词路径
        self.trainset_file_name = ""  # 训练集文件名 分词后
        self.tfidf_wordbag_file_name = ""  # 词包文件名

    # 对输入语料进行基本预处理，删除语料的换行符，并持久化。
    # 处理后在pos_path下建立与corpus_path相同的子目录和文件结构
    def preprocess(self):
        start = datetime.now()
        if (self.corpus_path == "" or self.pos_path == ""):
            print("corpus_path或pos_path不能为空")
            return
        dir_list = os.listdir(self.corpus_path)  # 获取每个目录下所有的文件
        for mydir in dir_list:
            class_path = self.corpus_path + '/' + mydir
            file_list = os.listdir(class_path)
            for name in file_list:
                file_name = class_path + "/" + name
                content = self.fileGBK2UTF(file_name)
                pos_dir = self.pos_path + "/" + mydir
                if not os.path.exists(pos_dir):
                    os.mkdir(pos_dir)
                file_write = open(pos_dir + "/" + name, 'w', encoding='utf-8')
                file_write.write(content)
                file_write.close()
        end = datetime.now()
        logger.info('预处理：完成GBK转成UTF。耗时：%s' %str((end-start).seconds))

    # 检测字符编码
    def detectCode(self, filename):
        detector = UniversalDetector()  # 初始化一个UniversalDetector对象
        f = open(filename, 'rb')
        for line in f:
            detector.feed(line)  # 逐行载入UniversalDetector对象中进行识别
            if detector.done:  # done为一个布尔值，默认为False，达到阈值时变为True
                break

        detector.close()  # 调用该函数做最后的数据整合
        f.close()
        codetype = detector.result['encoding']
        if codetype:
            print('****' + codetype + '****')
            if codetype.startswith('utf'):
                logger.info(filename + ' : codetype UTF-8')
                return 'utf'
            else:
                return 'gbk'
        else:
            return 'gbk'

    # GBK转成UTF
    def fileGBK2UTF(self, filename):
        typeCode = self.detectCode(filename)
        content = None
        if typeCode == 'gbk':
            try:
                with open(filename, 'rb')as file_obj:
                    content = file_obj.read().decode(encoding='gbk', errors='ignore')
            except Exception as e:
                logger.error(filename + ": " + e)
            finally:
                pass
        else:
            try:
                with open(filename, 'rb') as file_obj:
                    content = file_obj.read().decode(encoding='utf-8', errors='ignore')
            except Exception as e:
                logger.error(filename + ": " + e)
            finally:
                pass
        return content

    # 对预处理后语料进行分词,并持久化。
    # 处理后在segment_path下建立与pos_path相同的子目录和文件结构
    def segment(self):
        start = datetime.now()
        if (self.segment_path == "" or self.pos_path == ""):
            print("segment_path或pos_path不能为空")
            return
        global STOP_WORDS_LIST
        if not STOP_WORDS_LIST:
            file_read = open(STOP_WORDS_DIR, 'r', encoding='utf-8')
            STOP_WORDS_LIST = [w.strip() for w in file_read if w]
            file_read.close()

        dir_list = os.listdir(self.pos_path)
        # 获取每个目录下所有的文件
        for mydir in dir_list:
            class_path = self.pos_path + "/" + mydir  # 拼出分类子目录的路径
            file_list = os.listdir(class_path)  # 获取class_path下的所有文件
            for name in file_list:  # 遍历所有文件
                file_name = class_path + "/" + name  # 拼出文件名全路径
                print(file_name)
                file_read = open(file_name, 'r', encoding='utf-8')  # 打开一个文件
                raw_corpus = file_read.read()  # 读取未分词语料
                seg_corpus = [w.strip() for w in jieba.cut(raw_corpus) if
                              w not in STOP_WORDS_LIST and len(w.strip()) > 0]
                # 拼出分词后语料分类目录
                seg_dir = self.segment_path + "/" + mydir
                if not os.path.exists(seg_dir):  # 如果没有创建
                    os.makedirs(seg_dir)
                file_write = open(seg_dir + "/" + name, 'w', encoding='utf-8')  # 创建分词后语料文件，文件名与未分词语料相同
                file_write.write(" ".join(seg_corpus))  # 用空格将分词结果分开并写入到分词后语料文件中
                file_read.close()  # 关闭打开的文件
                file_write.close()  # 关闭写入的文件
        end = datetime.now()
        logger.info('分词成功完成。耗时：%s' %str((end-start).seconds))

    # 打包分词后训练语料
    def train_bag(self):
        start = datetime.now()
        if (self.segment_path == "" or self.model_path == "" or self.trainset_file_name == ""):
            print("segment_path或wordbag_path,trainset_name不能为空")
            return
            # 获取corpus_path下的所有子分类
        dir_list = os.listdir(self.segment_path)
        self.dataset_Bunch.target_name = dir_list
        # 获取每个目录下所有的文件
        for mydir in dir_list:
            class_path = self.segment_path + "/"+ mydir   # 拼出分类子目录的路径
            file_list = os.listdir(class_path)  # 获取class_path下的所有文件
            for file_path in file_list:  # 遍历所有文档
                file_name = class_path + "/"+file_path  # 拼出文件名全路径
                self.dataset_Bunch.filenames.append(file_name)  # 把文件路径附加到数据集中
                self.dataset_Bunch.label.append(self.dataset_Bunch.target_name.index(mydir))  # 把文件分类标签附加到数据集中
                file_read = open(file_name, 'r', encoding='utf-8')  # 打开一个文件
                seg_corpus = file_read.read()  # 读取语料
                self.dataset_Bunch.contents.append(seg_corpus)  # 构建分词文本内容列表
                file_read.close()
                # 词袋对象持久化
        if not os.path.exists(self.model_path ):
            os.mkdir(self.model_path)
        file_obj = open(self.model_path + "/" + self.trainset_file_name, "wb")
        pickle.dump(self.dataset_Bunch, file_obj)
        file_obj.close()
        end = datetime.now()
        logger.info('分词语料打包成功完成。耗时：%s' %str((end-start).seconds))


    # 计算训练语料的tfidf权值并持久化为词袋
    def tfidf_bag(self):
        start = datetime.now()
        if (self.model_path == "" or self.tfidf_wordbag_file_name == "" or self.stopword_path == ""):
            print("model_path，word_bag或stopword_path不能为空")
            return

        # 读取持久化后的训练集对象
        file_obj = open(self.model_path +"/"+ self.trainset_file_name, "rb")
        self.dataset_Bunch = pickle.load(file_obj)
        file_obj.close()
        # 定义词袋数据结构: tdm:tf-idf计算后词袋
        self.tfidf_wordbag_Bunch.target_name = self.dataset_Bunch.target_name
        self.tfidf_wordbag_Bunch.label = self.dataset_Bunch.label
        self.tfidf_wordbag_Bunch.filenames = self.dataset_Bunch.filenames
        # 构建语料
        corpus = self.dataset_Bunch.contents
        # 使用TfidfVectorizer初始化向量空间模型--创建词袋
        vectorizer = TfidfVectorizer(max_df=0.9)
        # 文本转为词频矩阵
        self.tfidf_wordbag_Bunch.tfidf = vectorizer.fit_transform(corpus)
        # 保存词袋词典文件
        self.tfidf_wordbag_Bunch.vocabulary = vectorizer.vocabulary_
        if not os.path.exists(self.model_path ):
            os.mkdir(self.model_path)
        # 创建词袋的持久化
        file_obj = open(self.model_path + "/" + self.tfidf_wordbag_file_name, "wb")
        pickle.dump(self.tfidf_wordbag_Bunch, file_obj)
        file_obj.close()
        end = datetime.now()
        logger.info('Tf-idf词袋创建成功。耗时：%s' %str((end-start).seconds))


    # 验证持久化结果：
    def verify_trainset(self):
        file_obj = open(self.model_path + '/' + self.trainset_file_name, 'rb')
        # 读取持久化后的对象
        self.dataset_Bunch = pickle.load(file_obj)
        file_obj.close()
        # 输出数据集包含的所有类别
        print(self.dataset_Bunch.target_name)
        # 输出数据集包含的所有类别标签数
        print(len(self.dataset_Bunch.label))
        # 输出数据集包含的文件内容数
        print(len(self.dataset_Bunch.contents))

    def verify_wordbag(self):
        file_obj = open(self.model_path + '/' + self.tfidf_wordbag_file_name, 'rb')
        # 读取持久化后的对象
        self.tfidf_wordbag_Bunch = pickle.load(file_obj)
        file_obj.close()
        # 输出数据集包含的所有类别
        print(self.tfidf_wordbag_Bunch.target_name)
        # 输出数据集包含的所有类别标签数
        print(len(self.tfidf_wordbag_Bunch.label))
        # 输出数据集包含的文件内容数
        print(self.tfidf_wordbag_Bunch.tfidf.shape)

    # 只进行tfidf权值计算：stpwrdlst:停用词表;myvocabulary:导入的词典
    def tfidf_value(self, test_data, myvocabulary):
        vectorizer = TfidfVectorizer(vocabulary=myvocabulary)
        return vectorizer.fit_transform(test_data)

    # 导出词袋模型：
    def load_wordbag(self):
        file_obj = open(self.model_path + '/' + self.tfidf_wordbag_file_name, 'rb')
        self.tfidf_wordbag_Bunch = pickle.load(file_obj)
        file_obj.close()

    # 导出训练语料集
    def load_trainset(self):
        file_obj = open(self.model_path + '/' + self.trainset_file_name, 'rb')
        self.dataset_Bunch = pickle.load(file_obj)
        file_obj.close()
