import numpy as np
import tensorflow as tf
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = ""
TEST_DATA_FILE = "train.csv"
TFIDF_VECTOR_FILE = "tdidf.pkl"
SENTENCE_VECTORS_FILE = "sentence_vectors.pkl"
test_data = pd.read_csv(TEST_DATA_FILE)
NUM_CLASSES = 6
NUM_FEATURES = 500

def vectorize_corpus_tf_idf():
    """ vectorizes the corpus using tf-idf
    """
    if os.path.isfile(TFIDF_VECTOR_FILE) and os.path.isfile(SENTENCE_VECTORS_FILE):
        with open(TFIDF_VECTOR_FILE, "rb") as fp:
            feature_dict = pickle.load(fp)
        with open(SENTENCE_VECTORS_FILE, "rb") as fp:
            sentence_vectors = pickle.load(fp)
        return feature_dict, sentence_vectors
    else:
        test_data = pd.read_csv(TEST_DATA_FILE)
        comments = test_data[['comment_text']].values.flatten()
        vectorizer = TfidfVectorizer(max_features=NUM_FEATURES)
        X = vectorizer.fit_transform(comments)
        idf = vectorizer.idf_
        feature_dict = dict(zip(vectorizer.get_feature_names(), idf))
        sentence_vectors = X.toarray()

        # save vectors and sentences
        with open(TFIDF_VECTOR_FILE, "wb") as fp:
            pickle.dump(feature_dict, fp)
        with open(SENTENCE_VECTORS_FILE, "wb") as fp:
            pickle.dump(sentence_vectors, fp)
        return feature_dict, sentence_vectors


if __name__ == "__main__":
    feature_dict, sentence_vectors = vectorize_corpus_tf_idf()
