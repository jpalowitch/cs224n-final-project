import pandas as pd
import csv
import os
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from project_utils import get_TDT_split
import pickle
GLOVE_DIRECTORY = "glove.6B"
GLOVE_EMBEDDINGS = "glove.6B.embeddings.pkl"
EMBEDDING_DIM = 50
TRAIN_DATA_FILE = "train.csv"
MAX_SENTENCE_LENGTH = 50
N_DIMENSIONS = 300
UNK_TOKEN = np.zeros(N_DIMENSIONS)

def vectorize_sentences_concat(df, embeddings):
    """ Vectorizes sentences by concatenating the GloVe representations of each word.
        Also adds padding if the sentence is smaller than the max length and cuts
        the sentence if it is too long.

    Args:
        df: sentences to vectorize
        embeddings: trained word vectors
    Returns:
        sentences: list of vectorized representation of words
        masks: boolean list of values to identify which tokens were added
    """
    sentences = []
    masks = []
    for sentence in df:
        if len(sentence) > MAX_SENTENCE_LENGTH:
            sentence = sentence[:(MAX_SENTENCE_LENGTH + 1)]
        mask = [True] * len(sentence)
        vectorized_sentence = []
        for word in sentence:
            glove_vector = embeddings.get(word)
            if glove_vector is not None:
                vectorized_sentence.append(glove_vector)
            else:
                vectorized_sentence.append(UNK_TOKEN)
        if len(sentence) < MAX_SENTENCE_LENGTH:
            addition_size = MAX_SENTENCE_LENGTH - len(sentence)
            mask_addition = [False] * addition_size
            word_addition = [UNK_TOKEN] * addition_size
            vectorized_sentence = vectorized_sentence + word_addition
            mask = mask + mask_addition
        sentences.append(vectorized_sentence)
        masks.append(mask)
    return sentences, masks

def vectorize_sentences_sum(df, embeddings):
    """ Vectorizes sentences by summing the GloVe representations of each word.

    Args:
        df: sentences to vectorize
        embeddings: trained word vectors
    Returns:
        sentences: list of vectorized representation of words
    """
    sentences = []
    for sentence in df:
        vectorized_sentence = np.zeros(N_DIMENSIONS)
        for word in sentence:
            glove_vector = embeddings.get(word)
            # only add if in embedding
            if glove_vector is not None:
                vectorized_sentence = np.add(vectorized_sentence, glove_vector)

        sentences.append(vectorized_sentence)
    return sentences

def get_tokenized_sentences(path="glove_tokenized_sentences.pkl", load_files=True):
    """ Returns an object containing the sentence vectors and word embeddings lookup
        for each dataset.

    Args:
        path: path to save the vectors to
        load_files: whether to use the saved values
    Returns:
        sentence_vectors: Object containing sentence vectors and tokenizer
        information for each dataset.
    """
    if os.path.isfile(path) and load_files:
        with open(path, "rb") as fp:
            sentence_vectors = pickle.load(fp)
        return sentence_vectors
    else:
        train, dev, test = get_TDT_split(pd.read_csv('train.csv').fillna(' '))
        # Create separate tokenizer for each set
        train_tokenizer = Tokenizer()
        dev_tokenizer = Tokenizer()
        test_tokenizer = Tokenizer()

        train_df = train[["comment_text"]].values.flatten()
        dev_df = dev[["comment_text"]].values.flatten()
        test_df = test[["comment_text"]].values.flatten()

        train_tokenizer.fit_on_texts(train_df)
        dev_tokenizer.fit_on_texts(dev_df)
        test_tokenizer.fit_on_texts(test_df)

        embeddings = get_word_vectors()
        train_vectors_concat, train_masks = vectorize_sentences_concat(train_df, embeddings)
        dev_vectors_concat, dev_masks = vectorize_sentences_concat(dev_df, embeddings)
        test_vectors_concat, test_masks = vectorize_sentences_concat(test_df, embeddings)

        sentence_vectors = {
            "train": {
                "vectors_sum": vectorize_sentences_sum(train_df, embeddings),
                "vectors_concat": train_vectors_concat,
                "masks": train_masks,
                "word_index": train_tokenizer.word_index
            },
            "dev": {
                "vectors_sum": vectorize_sentences_sum(dev_df, embeddings),
                "vectors_concat": dev_vectors_concat,
                "masks": dev_masks,
                "word_index": dev_tokenizer.word_index
            },
            "test": {
                "vectors_sum": vectorize_sentences_sum(test_df, embeddings),
                "vectors_concat": test_vectors_concat,
                "masks": test_masks,
                "word_index": test_tokenizer.word_index
            }
        }

        with open(path, "wb") as fp:
            pickle.dump(sentence_vectors, fp)
        return sentence_vectors

def get_word_vectors(path=GLOVE_EMBEDDINGS, load_files=True):
    """ Fetches GloVe word embeddings.

    Args:
        path: file path for saving the word embeddings
        load_files: whether to load the files
    Returns:
        embeddings_index: the word embeddings
    """
    if os.path.isfile(path) and load_files:
        with open(path, "rb") as fp:
            embeddings_index = pickle.load(fp)
        return embeddings_index
    else:
        embeddings_index = {}
        # glove_vectors = open(os.path.join(GLOVE_DIRECTORY, "glove.6B.50d.txt"))
        glove_vectors = open("glove.42B.300d.txt", "rb")
        for line in glove_vectors:
            values = line.split()
            word = values[0]
            coefficients = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefficients
        glove_vectors.close()
        with open(path, "wb") as fp:
            pickle.dump(embeddings_index, fp)
        return embeddings_index

def get_embedding_matrix(path="embeddings.pkl", data_set="train", load_files=True):
    """ Creates embedding matrix for data set.

    Args:
        path: file path for saved embeddings
        data_set: one of "train", "dev", or "test"
        load_files: whether to load saved copies of the files
    Returns:
        embedding_matrix: embedding matrix
    """
    print "Fetching embedding matrix"
    embedding_path = data_set + "_" + path
    if os.path.isfile(embedding_path) and load_files:
        with open(embedding_path, "rb") as fp:
            embeddings = pickle.load(fp)
        return embeddings
    else:
        embeddings = get_word_vectors()
        sentence_vectors = get_tokenized_sentences().get(data_set)
        embedding_matrix = np.zeros((len(sentence_vectors.get("word_index")) + 1, EMBEDDING_DIM))
        for word, idx in sentence_vectors.get("word_index").items():
            embedding_vector = embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
        with open(embedding_path, "wb") as fp:
            pickle.dump(embedding_matrix, fp)
        return embedding_matrix


if __name__ == "__main__":
    # embeddings = get_embedding_matrix(load_files=False)
    # sentences = get_tokenized_sentences()
    # print sentences.get("train").get("vectors_concat")
