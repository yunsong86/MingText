import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk.data
from nltk.corpus import stopwords
from gensim.models import word2vec


def review_to_text(review, remove_stopwords):
    raw_text = BeautifulSoup(review, 'lxml').get_text()

    letters = re.sub('[^a-zA-Z]', ' ', raw_text)

    words = letters.lower().split()

    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]

    return words


def review_to_sentences(review, tokenizer):
    raw_sentences = tokenizer.tokenize(review.strip())

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_text(raw_sentence, False))

    return sentences


if __name__ == '__main__':
    unlabeled_train = pd.read_csv('/home/ys/PycharmProjects/book1/Datasets/IMDB/unlabeledTrainData.tsv', delimiter='\t',
                                  quoting=3)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    corpora = []

    for review in unlabeled_train['review']:
        print(review)
        corpora += review_to_sentences(review, tokenizer)

    # Set values for various parameters
    num_features = 100  # Word vector dimensionality
    min_word_count = 20  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 6  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    from gensim.models import word2vec

    print("Training model...")

    model = word2vec.Word2Vec(corpora, workers=num_workers, size=num_features, min_count=min_word_count, window=context,
                              sample=downsampling)

    model.init_sims(replace=True)

    model_name = "/home/ys/PycharmProjects/book1/Datasets/IMDB/features100_minwords20_context6"
    model.save(model_name)


