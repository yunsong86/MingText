from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from  Log import *

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")

    nwords = 0.

    index2word_set = set(model.wv.index2word)

    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])

    featureVec = np.divide(featureVec, nwords)
    # print(featureVec)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)

        counter += 1
    print(reviewFeatureVecs)
    return reviewFeatureVecs


def review_to_text(review, remove_stopwords):
    raw_text = BeautifulSoup(review, 'lxml').get_text()

    letters = re.sub('[^a-zA-Z]', ' ', raw_text)

    words = letters.lower().split()

    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]

    return words


if __name__ == "__main__":
    num_features = 100  # Word vector dimensionality

    model = Word2Vec.load("/home/ys/PycharmProjects/book1/Datasets/IMDB/features100_minwords20_context6")
    model.most_similar("man")

    train = pd.read_csv('/home/ys/PycharmProjects/book1/Datasets/IMDB/labeledTrainData.tsv', delimiter='\t')

    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(review_to_text(review, remove_stopwords=True))

    trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

    test = pd.read_csv('/home/ys/PycharmProjects/book1/Datasets/IMDB/testData.tsv', delimiter='\t')
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_text(review, remove_stopwords=True))

    testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

    gbc = GradientBoostingClassifier()

    params_gbc = {'n_estimators': [10, 100, 500], 'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [2, 3, 4]}
    gs = GridSearchCV(gbc, params_gbc, cv=4, n_jobs=-1, verbose=1)

    y_train = train['sentiment']

    gs.fit(trainDataVecs, y_train)

    logger.info(gs.best_score_)
    logger.info(gs.best_params_)

    result = gs.predict(testDataVecs)
    # Write the test results
    # output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    # output.to_csv("/home/ys/PycharmProjects/book1/Datasets/IMDB/submission_w2v.csv", index=False, quoting=3)
