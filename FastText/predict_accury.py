# !/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import fasttext  
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  
classifier = fasttext.supervised("D:/UbunutWin/corpus/fastextcorpus/news_fasttext_train.txt","D:/UbunutWin/corpus/fastextcorpus/news_fasttext.model",label_prefix="__label__")
#load训练好的模型  
classifier = fasttext.load_model('D:/UbunutWin/corpus/fastextcorpus/news_fasttext.model.bin', label_prefix='__label__')
result = classifier.test("D:/UbunutWin/corpus/fastextcorpus/news_fasttext_test.txt")
print(result.precision)  
print(result.recall)  
labels_right = []  
texts = []  
with open("D:/UbunutWin/corpus/fasttestcorpus/news_fasttext_test.txt", encoding='utf-8') as fr:
    lines = fr.readlines()  
for line in lines:  
    labels_right.append(line.split("\t")[1].rstrip().replace("__label__",""))  
    texts.append(line.split("\t")[0])  
#     print labels  
#     print texts  
#     break  
labels_predict = [e[0] for e in classifier.predict(texts)] #预测输出结果为二维形式  
# print labels_predict  
text_labels = list(set(labels_right))  
text_predict_labels = list(set(labels_predict))  
print(text_predict_labels)  
print(text_labels)  
A = dict.fromkeys(text_labels,0)  #预测正确的各个类的数目  
B = dict.fromkeys(text_labels,0)   #测试数据集中各个类的数目  
C = dict.fromkeys(text_predict_labels,0) #预测结果中各个类的数目  
for i in range(0,len(labels_right)):  
    B[labels_right[i]] += 1  
    C[labels_predict[i]] += 1  
    if labels_right[i] == labels_predict[i]:  
        A[labels_right[i]] += 1  
print(A )  
print(B)  
print( C)  
#计算准确率，召回率，F值  
for key in B:  
    p = float(A[key]) / float(B[key])  
    r = float(A[key]) / float(C[key])  
    f = p * r * 2 / (p + r)  
    print ("%s:\tp:%f\t%fr:\t%f" % (key,p,r,f)) 