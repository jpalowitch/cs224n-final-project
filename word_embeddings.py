import pandas as pd
import csv
import os
import numpy as np
from project_utils import get_TDT_split, tokenize
import pickle
import tensorflow as tf
from project_utils import getopts
from sys import argv
from glove import generate_embeddings_all, get_pretrained_glove_vectors, embedding_size
from scipy.spatial.distance import cosine

EMBEDDING_DIM = embedding_size
TRAIN_DATA_FILE = "train.csv"
MAX_SENTENCE_LENGTH = 50
N_DIMENSIONS = embedding_size
UNK_TOKEN = np.zeros(N_DIMENSIONS)
use_local_glove_vectors = None
# Link for GloVe vectors: https://nlp.stanford.edu/projects/glove/

# use tensorflow hosted versions
Tokenizer = tf.keras.preprocessing.text.Tokenizer
text_to_word_sequence = tf.keras.preprocessing.text.text_to_word_sequence

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

def vectorize_sentence_average(word_index_reverse, pretrained_embeddings, sequences, use_local=False, local_embeddings=None):
    """ Vectorizes sentences by summing the GloVe representations of each word.

    Args:
        df: sentences to vectorize
        embeddings: trained word vectors
        use_local: whether to use the embeddings trained on the corpus
        local_embeddings: embeddings trained on the corpus
    Returns:
        sentences: single vector representation of the sentence
    """
    print "Vectorizing sentences"
    sentences = []
    # use command line arg if defined
    if use_local_glove_vectors is not None:
        use_local = use_local_glove_vectors

    if use_local == True:
        print "Using local GloVe vectors!"

    for idx, token_ids in enumerate(sequences):
        if idx % 3000 == 0:
            print "On line: {}".format(idx)
        vectorized_sentence = np.zeros((N_DIMENSIONS,))
        added = False
        for token in token_ids:
            if use_local == True:
                vectorized_sentence = vectorized_sentence + local_embeddings[token]
                added = True
            else:
                word = word_index_reverse[token]
                glove_vector = pretrained_embeddings.get(word)
                # only add if in embedding
                if glove_vector is not None:
                    vectorized_sentence = vectorized_sentence + glove_vector
                    added = True

        # Only average if valid tokens added to sentence
        if added:
            vectorized_sentence = np.divide(vectorized_sentence, float(len(token_ids)))
        sentences.append(vectorized_sentence)
    print "Done"
    return sentences

def get_tokenized_sentences(path="data/glove_tokenized_sentences.pkl", load_files=True, use_local=False):
    """ Returns an object containing the sentence vectors and word embeddings lookup
        for each dataset.

    Args:
        path: path to save the vectors to
        load_files: whether to use the saved values
        use_local: whether to use locally trained GloVe word embeddings
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
        tokenizer = Tokenizer()

        train_df = train[["comment_text"]].values.flatten()
        dev_df = dev[["comment_text"]].values.flatten()
        test_df = test[["comment_text"]].values.flatten()

        tokenizer.fit_on_texts(pd.read_csv('train.csv').fillna(' ')[["comment_text"]].values.flatten())

        train_sequences = tokenizer.texts_to_sequences(train_df)
        dev_sequences = tokenizer.texts_to_sequences(dev_df)
        test_sequences = tokenizer.texts_to_sequences(test_df)

        embeddings = get_pretrained_glove_vectors()
        local_embeddings  = read_local_vectors()

        # maps word indices to words so that it can be read from the embeddings matrix
        word_index_reverse = {v:k for k, v in tokenizer.word_index.items()}

        sentence_vectors = {
            "train": {
                "vectors": vectorize_sentence_average(word_index_reverse, embeddings, train_sequences, \
                                                        use_local=use_local, local_embeddings=local_embeddings),
                "word_index": tokenizer.word_index,
                "sequences": train_sequences
            },
            "dev": {
                "vectors": vectorize_sentence_average(word_index_reverse, embeddings, dev_sequences, \
                                                        use_local=use_local, local_embeddings=local_embeddings),
                "word_index": tokenizer.word_index,
                "sequences": dev_sequences
            },
            "test": {
                "vectors": vectorize_sentence_average(word_index_reverse, embeddings, test_sequences, \
                                                        use_local=use_local, local_embeddings=local_embeddings),
                "word_index": tokenizer.word_index,
                "sequences": test_sequences
            }
        }

        with open(path, "wb") as fp:
            pickle.dump(sentence_vectors, fp)
        return sentence_vectors

def get_embedding_matrix_and_sequences(path="embeddings.pkl", data_set="train", load_files=False, use_local=False):
    """ Creates embedding matrix for data set and returns embedding matrix and
        an ordered list of word index sequences.

    Args:
        path: file path for saved embeddings
        data_set: one of "train", "dev", or "test"
        load_files: whether to load saved copies of the files
        use_local: whether to use locally trained vectors
    Returns:
        embedding_matrix: embedding matrix mappings words to GloVe vectors
        sequences: ordered list of sentences where each element is an index in
                   the embedding matrix
    """
    embedding_path = data_set + "_" + path
    if os.path.isfile(embedding_path) and load_files:
        with open(embedding_path, "rb") as fp:
            embeddings_and_sequences = pickle.load(fp)
        return embeddings_and_sequences.get("embeddings"), embeddings_and_sequences.get("sequences")
    else:
        sentence_vectors = get_tokenized_sentences().get(data_set)
        # override with command line arg if present
        if use_local_glove_vectors is not None:
            use_local = use_local_glove_vectors
        # check whether to look for vectors trained on corpus
        if use_local:
            print "Using locally trained vectors"
            embedding_matrix = read_local_vectors()

        else:
            embeddings = get_pretrained_glove_vectors()
            embedding_matrix = np.zeros((len(sentence_vectors.get("word_index")) + 1, N_DIMENSIONS))
            # map word indices to GloVe vectors
            for word, idx in sentence_vectors.get("word_index").items():
                embedding_vector = embeddings.get(word)
                if embedding_vector is not None:
                    embedding_matrix[idx] = embedding_vector

        # get sequences and save
        sequences = sentence_vectors.get("sequences")
        with open(embedding_path, "wb") as fp:
            embeddings_and_sequences = {
                "embeddings": embedding_matrix,
                "sequences": sequences
            }
            pickle.dump(embeddings_and_sequences, fp)
        return embedding_matrix, sequences

def read_local_vectors(path="data/all_100_10000_25_0.05_5_True_embeddings.pkl"):
    """ Returns the GloVe vectors trained for the dataset

    Args:
        data_set: one of train, dev, test

    Returns:
        embeddings: GloVe embeddings
    """
    if os.path.isfile(path):
        print "Reading local embeddings: {}".format(path)
        with open(path, "rb") as fp:
            embeddings = pickle.load(fp)
        print "Done"
        return embeddings
    else:
        print "Could not find embeddings with path {}, generating new ones".format(path)
        embeddings = generate_embeddings_all()
        print "Done"
        return embeddings

## UTILS
def generate_local_tsne(words=None, corpus=None, threshold=2000):
    """ Generates a scatter plot of the locally trained word vectors

    Args:
        words: list of words to visualize in the plot
        corpus: corpus to use for the visualization
        threshold: top words to include
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    if corpus is None:
        corpus = pd.read_csv('train.csv').fillna(' ')[["comment_text"]].values.flatten()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    word_index_reverse = {v:k for k, v in tokenizer.word_index.items()}
    embeddings = read_local_vectors()

    # train on top 2000 most occurring words - rank == index for the tokenizer
    vectors = embeddings[:threshold]
    # generate tsne
    X_embedded = TSNE(n_components=2, n_iter=5000, perplexity=20, init="pca").fit_transform(vectors)
    # Only plot words interested in
    x_chosen = []
    y_chosen = []
    labels_chosen = []
    for word in words:
        word_idx = tokenizer.word_index.get(word)
        if word_idx >= threshold:
            print "Word not in top {} words: {}".format(threshold, word)
        else:
            row = X_embedded[word_idx]
            x_chosen.append(row[0])
            y_chosen.append(row[1])
            labels_chosen.append(word)

    plt.scatter(x_chosen, y_chosen)

    for i in range(len(labels_chosen)):
        plt.annotate(labels_chosen[i], (x_chosen[i], y_chosen[i]))
    plt.show()

def test_plot():
    words = ["man", "boy", "woman", "girl", "fuck", "hate", "asshole", "penis", "dick", "bitch", "the", "a"]
    generate_local_tsne(words=words)

def compare_words(words, num_words=10000, embedding_size=100):
    """ Compares pairs of words using the locally trained vectors as well as
        to their glove counterparts

    Args:
        words: List of tuples of (word, word)
    """
    # change this if using different local glove embeddings than
    # "data/all_50_10000_2_embeddings.pkl"
    if embedding_size == 100:
        path="data/all_100_10000_15_0.05_5_embeddings.pkl"
    else:
        path="data/all_50_10000_2_embeddings.pkl"

    # fit tokenizer to corpus
    df = pd.read_csv('train.csv').fillna(' ')[["comment_text"]].values.flatten()
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(df)

    # fetch GloVe vectors
    glove_embeddings = get_pretrained_glove_vectors()
    local_embeddings = read_local_vectors(path=path)

    for (word1, word2) in words:
        print
        print 'word1: {} word2: {}'.format(word1, word2)
        word1_vector = local_embeddings[tokenizer.word_index.get(word1)]
        word2_vector = local_embeddings[tokenizer.word_index.get(word2)]
        diff = cosine(word1_vector, word2_vector)
        glove_diff = cosine(glove_embeddings.get(word1), glove_embeddings.get(word2))
        print 'local difference: {}'.format(diff)
        print 'glove difference: {}'.format(glove_diff)
        print
        print '--------GloVe comparison--------'
        glove_vector = glove_embeddings.get(word1)
        if glove_vector is not None:
            glove_diff = cosine(word1_vector, glove_vector)
            print 'Word 1 GloVe difference: {}'.format(glove_diff)
        else:
            print 'Could not find GloVe vector for {}'.format(word1)

        glove_vector = glove_embeddings.get(word2)
        if glove_vector is not None:
            glove_diff = cosine(word2_vector, glove_vector)
            print 'Word 2 GloVe difference: {}'.format(glove_diff)
        else:
            print 'Could not find GloVe vector for {}'.format(word2)

def test_compare_words(num_words):
    arr = [("man", "boy"), ("woman", "girl"), ("death", "dead"), ("eat", "ate"), ("you", "i"), ("he", "she"), ("him", "her")]
    compare_words(arr, num_words=num_words)

def test_data_set_glove_vectors():
    train, _, _ = get_TDT_split(pd.read_csv('train.csv').fillna(' '))
    train_df = train[["comment_text"]].values.flatten()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df)
    embeddings = read_local_vectors()
    assert len(embeddings) == (len(tokenizer.word_index.keys()) + 1)
    print "GloVe vector dimensions are correct"

def test_glove_vectors():
    df = pd.read_csv('train.csv').fillna(' ')[["comment_text"]].values.flatten()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df)
    embeddings = read_local_vectors()
    assert len(embeddings) == (len(tokenizer.word_index.keys()) + 1)
    print "GloVe vector dimensions are correct"

def test_get_embedding_matrix_and_sequences():
    embeddings, sequences = get_embedding_matrix_and_sequences()
    print 'embedding 1: {}'.format(embeddings[1])
    print 'sequence 1: {}'.format(sequences[1])
    sequence = sequences[1]
    for word in sequence:
        print 'word: {} embedding: {}'.format(word, embeddings[word])

def test_get_tokenized_sentences_local_embeddings():
    sentences = get_tokenized_sentences(load_files=False, use_local=True)
    train_vectors = sentences.get("train").get("vectors")
    print train_vectors[0]

if __name__ == "__main__":
    myargs = getopts(argv)
    if "-em" in myargs:
        use_local_glove_vectors = bool(myargs["-em"])

    if "-test" in myargs:
        if myargs["-test"] == "matrix":
            test_get_embedding_matrix_and_sequences()
        elif myargs["-test"] == "sentences":
            test_get_tokenized_sentences_local_embeddings()
        elif myargs["-test"] == "glove":
            test_glove_vectors()
        elif myargs["-test"] == "words":
            test_compare_words(100000)
        elif myargs["-test"] == "plot":
            test_plot()
