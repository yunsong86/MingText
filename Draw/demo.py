#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
@author: PANYUNSONG
@file: demo.py
@time: 17-10-13 下午6:43
@desc: python3.6
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 0、导入数据集
df = pd.read_excel('./sampledata.xlsx', 'Sheet1')
# 1、直方图
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(df['Age'], bins=7)
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Employee')
plt.show()



# 2、箱线图
fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot(df['Age'])
plt.show()



# 3、小提琴图
sns.violinplot(df['Age'], df['Gender'])
sns.despine()
plt.show()



# 4、条形图
var = df.groupby('Gender').Sales.sum()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel('Gender')
ax1.set_ylabel('Sum of Sales')
ax1.set_title('Gender wise Sum of Sales')
var.plot(kind='bar')
plt.show()


# 5、折线图
var = df.groupby('BMI').Sales.sum()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('BMI')
ax.set_ylabel('Sum of Sales')
ax.set_title('BMI wise Sum of Sales')
var.plot(kind='line')
plt.show()

# 6、堆积柱形图
var = df.groupby(['BMI', 'Gender']).Sales.sum()
var.unstack().plot(kind='bar', stacked=True, color=['red', 'blue'])
plt.show()



# 7、散点图
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(df['Age'], df['Sales'])
plt.show()


# 8、气泡图
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(df['Age'], df['Sales'], s=df['Income'])  # 第三个变量表明根据收入气泡的大小
plt.show()


# 9、饼图
var = df.groupby(['Gender']).sum().stack()
temp = var.unstack()
type(temp)
x_list = temp['Sales']
label_list = temp.index
plt.axis('equal')
plt.pie(x_list, labels=label_list, autopct='%1.1f%%')
plt.title('expense')
plt.show()



# 10、热图
data = np.random.rand(4, 2)
rows = list('1234')
columns = list('MF')
fig, ax = plt.subplots()
ax.pcolor(data, cmap=plt.cm.Reds, edgecolor='k')
ax.set_xticks(np.arange(0, 2)+0.5)
ax.set_yticks(np.arange(0, 4)+0.5)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.set_xticklabels(columns, minor=False, fontsize=20)
ax.set_yticklabels(rows, minor=False, fontsize=20)
plt.show()

