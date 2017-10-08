# 从sklearn.datasets里导入20类新闻文本数据抓取器。
from sklearn.datasets import fetch_20newsgroups
# 从互联网上即时下载新闻样本,subset='all'参数代表下载全部近2万条文本存储在变量news中。
news = fetch_20newsgroups(subset='all')

# 从sklearn.cross_validation导入train_test_split模块用于分割数据集。
from sklearn.model_selection import train_test_split
# 对news中的数据data进行分割，25%的文本用作测试集；75%作为训练集。
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
# 从sklearn.feature_extraction.text里分别导入TfidfVectorizer。
from sklearn.feature_extraction.text import TfidfVectorizer
# 采用默认的配置对TfidfVectorizer进行初始化（默认配置不去除英文停用词），并且赋值给变量tfidf_vec。
tfidf_vec = TfidfVectorizer()

# 使用tfidf的方式，将原始训练和测试文本转化为特征向量。
X_tfidf_train = tfidf_vec.fit_transform(X_train)
X_tfidf_test = tfidf_vec.transform(X_test)

# 依然使用默认配置的朴素贝叶斯分类器，在相同的训练和测试数据上，对新的特征量化方式进行性能评估。
from sklearn.naive_bayes import MultinomialNB

mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(X_tfidf_train, y_train)
print('The accuracy of classifying 20newsgroups with Naive Bayes (TfidfVectorizer without filtering stopwords):', mnb_tfidf.score(X_tfidf_test, y_test))
y_tfidf_predict = mnb_tfidf.predict(X_tfidf_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_tfidf_predict, target_names = news.target_names))
