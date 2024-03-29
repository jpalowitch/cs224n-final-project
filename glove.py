from project_utils import tokenize as word_tokenizer
import numpy as np
from collections import defaultdict
import tensorflow as tf
import os
from project_utils import get_TDT_split, get_development_data, getopts
import pandas as pd
import pickle
from sys import argv

# use tensorflow hosted version of Tokenizer
Tokenizer = tf.keras.preprocessing.text.Tokenizer

glove_directory = "glove.6B"
glove_embeddings_path = "data/glove.6B.embeddings.pkl"

# Global vars set by command line arguments
batch_size = 512
num_epochs = 2
# default is 50 (if unspecified)
embedding_size = 100
device = "/cpu:0"
# detault is 100000 (if unspecified)
num_words=10000
# default is 0.05 (if unspecified)
learning_rate = 0.05
# default is 10 (if unspecified)
window_size = 10
# controls whether to initialize vectors with pretrained GloVe embeddings
initialize_with_pretrained = True

def build_coccurrence_matrix(corpus, min_frequency=2):
    """ Builds a cooccurrence matrix as a dictionary.

    Args:
        corpus: list of sentences
        window_size: size of the window to consider word frequencies in
        min_frequency: minimum frequency of a word to keep

    Returns:
        cooccurrence_matrix: dictionary of form {(center_word_index, context_word_index):count}
        tokenizer: word tokenizer to fetch information for word frequency and other values
    """
    # train tokenizer on corpus
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(corpus)

    # dict of {token_index: word}
    word_index_reverse = {v:k for k, v in tokenizer.word_index.items()}

    cooccurrence_matrix_unfiltered = defaultdict(float)
    sequences = tokenizer.texts_to_sequences(corpus)

    for idx, token_ids in enumerate(sequences):
        if idx % 1000 == 0:
            print "On line: {}".format(idx)
        # print 'sequence: {}'.format(token_ids)
        # v represents the center word; u is the context word vector
        for v_idx, v in enumerate(token_ids):
            # separate contexts for left side to use for both, because symmetry
            left_context_words = token_ids[max(0, v_idx - window_size) : v_idx]
            right_context_words = token_ids[(v_idx + 1) : min(len(token_ids), v_idx + window_size + 1)]
            left_context_size = len(left_context_words)
            right_context_size = len(right_context_words)

            # left context word
            for u_idx, u in enumerate(left_context_words):
                distance = left_context_size - u_idx
                cooccurrence_matrix_unfiltered[(v, u)] += (1.0 / float(distance))

            # right context words
            for u_idx, u in enumerate(right_context_words):
                distance = u_idx + 1
                cooccurrence_matrix_unfiltered[(v, u)] += (1.0 / float(distance))
    print "Completed counting occurences in {} iterations".format(idx)
    if min_frequency > 1:
        # matrix of (v, u) : count
        cooccurrence_matrix = defaultdict(float)
        for (v, u), count in cooccurrence_matrix_unfiltered.items():
            v_actual = word_index_reverse[v]
            u_actual = word_index_reverse[u]
            # print 'v: {} u: {} count: {}'.format(v, u, count)
            if tokenizer.word_counts[v_actual] >= min_frequency and tokenizer.word_counts[u_actual] >= min_frequency:
                 cooccurrence_matrix[(v, u)] = count

        return cooccurrence_matrix, tokenizer
    else:
        return cooccurrence_matrix_unfiltered, tokenizer


def build_graph_and_train(cooccurrence_matrix, vocab_size, scope, tokenizer, save_partial=False):
    """ Builds and trains a tensorflow model for creating GloVe vectors based off
        of a cooccurrence_matrix.

    Args:
        cooccurrence_matrix: dictionary of form {(input_word_index, context_word_index) : count}
        vocab_size: size of vocabulary
        scope: variable scope for tensors
        tokenizer: the tokenizer for the vocabulary
        save_partial: whether to store partially trained embeddings during
                      training time

    Returns:
        embeddings: (vocab_size x embedding_size) shape matrix of word vectors
    """
    # model params
    alpha = 0.75

    # upper bound for words that cooccur frequently
    x_ij_max = 100.0

    with tf.device(device):
        with tf.variable_scope(scope):
            # weights for input (i) and context (j) vectors
            # tokens are indexed at 1, so need shape needs to be vocab_size + 1
            if initialize_with_pretrained:
                W_i = tf.get_variable("W_i", shape=[vocab_size + 1, embedding_size], initializer=glove_pretrained_initializer(vocab_size, embedding_size, tokenizer))
                W_j = tf.get_variable("W_j", shape=[vocab_size + 1, embedding_size], initializer=glove_pretrained_initializer(vocab_size, embedding_size, tokenizer))
            else:
                W_i = tf.get_variable("W_i", initializer=tf.random_uniform([vocab_size + 1, embedding_size], 1.0, -1.0))
                W_j = tf.get_variable("W_j", initializer=tf.random_uniform([vocab_size + 1, embedding_size], 1.0, -1.0))
            # each word has a scalar weight for when it is a center or context word
            B_i = tf.get_variable("input_bias", initializer=tf.random_uniform([vocab_size + 1], 1.0, -1.0))
            B_j = tf.get_variable("context_bias", initializer=tf.random_uniform([vocab_size + 1], 1.0, -1.0))

        # add placeholders for indices for the input and output words and cooccurrence for (v, u)
        i = tf.placeholder(tf.int32, shape=(batch_size), name="input_batch")
        j = tf.placeholder(tf.int32, shape=(batch_size), name="output_batch")
        X_ij = tf.placeholder(tf.float32, shape=(batch_size), name="context_batch")

        # weights and biases
        w_i = tf.nn.embedding_lookup(W_i, i)
        w_j = tf.nn.embedding_lookup(W_j, j)
        b_i = tf.nn.embedding_lookup(B_i, i)
        b_j = tf.nn.embedding_lookup(B_j, j)

        # actual math
        # f (X_ij): map the function count < max_count ? (count/max_count)^alpha : 1 onto each element
        f = tf.map_fn(lambda x_ij: tf.cond(x_ij < x_ij_max, lambda: tf.pow(tf.divide(x_ij, x_ij_max), alpha), lambda: tf.cast(1.0, tf.float32)), X_ij)

        # w_i * w_j + b_i + b_j + log(X_ij)
        # the weights are row vectors so need to calculate outer product
        w_j_transpose = tf.transpose(w_j)
        inner_value = tf.matmul(w_i, w_j_transpose) + b_i + b_j - tf.log(X_ij)

        # loss: f(X_ij) * [(w_i * w_j + b_i + b_j + log(X_ij))^2]
        f_expanded = tf.expand_dims(f, 1)
        loss = f * tf.pow(inner_value, 2)

        # reduce for entire batch
        total_loss = tf.reduce_sum(loss)

        # treating final word vectors as i + j
        combined_embeddings = W_i + W_j

    # set optimizer
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

    # train
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(num_epochs):
            if epoch % 5 == 0:
                print "On epoch: {}".format(epoch)
            idx = 0
            minibatches = get_cooccurrence_batches(cooccurrence_matrix, batch_size)
            for input_batch, context_batch, count_batch in minibatches:
                if idx % 1000 == 0:
                    print "On iteration: {}".format(idx)

                feed_dict = {
                    i: input_batch,
                    j: context_batch,
                    X_ij: count_batch
                }
                sess.run(optimizer, feed_dict=feed_dict)
                idx += 1

            # save partially trained embedddings
            partial_embeddings = combined_embeddings.eval()
            full_path = "data/" + "partial_all" + "_" + str(embedding_size) + "_" +  \
                str(num_words) + "_" + str(epochs) + "_" + str(learning_rate) + \
                "_" + str(window_size) + "_" + str(initialize_with_pretrained) + "_" + path
            with open(full_path, "wb") as fp:
                pickle.dump(embeddings, fp)
        embeddings = combined_embeddings.eval()
        sess.close()
    return embeddings

def get_cooccurrence_matrix(path="data/cooccurrence.pkl", load_files=True):
    """ Builds and retuns the cooccurrence matrices for the train, dev, and test sets

    Args:
        path: path for the pkl file
        load_files: whether to load previosly generated matrices or build new ones

    Returns:
        matrix_and_tokenizer: dict that holds the cooccurrence matrix and the tokenizer
    """
    if os.path.isfile(path) and load_files:
        with open(path, "rb") as fp:
            matrix_and_tokenizer = pickle.load(fp)
        return matrix_and_tokenizer
    else:
        df = pd.read_csv('train.csv').fillna(' ')[["comment_text"]].values.flatten()
        matrix, tokenizer = build_coccurrence_matrix(df)

        matrix_and_tokenizer = {
            "matrix": matrix,
            "tokenizer": tokenizer
        }

        with open(path, "wb") as fp:
            pickle.dump(matrix_and_tokenizer, fp)
        return matrix_and_tokenizer

def get_cooccurrence_batches(cooccurrence_matrix, batch_size, shuffle=True):
    """ Generates a minibatch of data from a cooccurence matrix.

    Args:
        cooccurrence_matrix: dict of form {(center_index, context_index): count}
        batch_size: size of minibatch
        shuffle: whether to randomly shuffle the batch

    Returns:
        i: minibatch of input indices
        j: minibatch of context indices
        count: minibatch of X_ij counts
    """
    np_keys = np.array(cooccurrence_matrix.keys())
    np_values = np.array(cooccurrence_matrix.values())
    dimension = np_keys.shape[0]
    if shuffle:
        indices = np.arange(dimension)
        np.random.shuffle(indices)
    for i in range(0, dimension - batch_size + 1, batch_size):
        if shuffle:
            batch = indices[i:(i + batch_size)]
        else:
            batch = slice(i, i + batch_size)

        pairs = np_keys[batch]
        X_ij = np_values[batch]
        i = [pair[0] for pair in pairs]
        j = [pair[1] for pair in pairs]
        yield i, j, X_ij

def generate_embeddings_all(load_files=False, path="embeddings.pkl"):
    """ Generates embeddings of matrix based off of vocabulary

    Args:
        load_files: whether to load previously saved files
        path: end of file name for embeddings

    Returns:
        embeddings: list of word embeddings
    """
    full_path = "data/" + "all" + "_" + str(embedding_size) + "_" +  \
        str(num_words) + "_" + str(num_epochs) + "_" + str(learning_rate) + \
        "_" + str(window_size) + "_" + str(initialize_with_pretrained) + "_" + path

    print "path:"
    print full_path
    if os.path.isfile(full_path) and load_files:
        with open(full_path, "rb") as fp:
            embeddings = pickle.load(fp)
        return embeddings
    else:
        info = get_cooccurrence_matrix()
        tokenizer = info.get("tokenizer")
        matrix = info.get("matrix")
        vocab_size = len(tokenizer.word_index.keys())
        embeddings = build_graph_and_train(matrix, vocab_size, "all", tokenizer)
        with open(full_path, "wb") as fp:
            pickle.dump(embeddings, fp)
        return embeddings

def glove_pretrained_initializer(vocab_size, embedding_size, tokenizer, \
                                 load_files=True, path="pretrained_initializer.pkl"):
    """ Creates a word embeddings matrix initialized with pretrained GloVe
        vectors.

    Args:
        vocab_size: size of the vocabulary
        embedding_size: size of each word embedding
        tokenizer: tokenizer that holds information about the vocabulary
        load_files: whether to load a previously initialized matrix
        path: end of the path for the saved file

    Returns:
        init: a tensor of shape (vocab_size + 1, embedding_size) that is
              initialized with GloVe pretrained vectors
    """
    full_path = "data/" + str(embedding_size) + "_" + path
    if os.path.isfile(full_path) and load_files:
        print "Reading GloVe pretrained initializer"
        with open(full_path, "rb") as fp:
            init = pickle.load(fp)
        print "Done"
        return init
    else:
        print "Generating GloVe pretrained initializer"
        glove_vectors = get_pretrained_glove_vectors()

        # initialize embeddings randomly
        W =  np.random.uniform(low=-1.0, high=1.0, size=(vocab_size + 1, embedding_size))

        for word, idx in tokenizer.word_index.items():
            glove_embedding = glove_vectors.get(word)
            if glove_embedding is not None:
                W[idx] = glove_embedding

        # convert to tensor
        init = tf.constant_initializer(W)

        # save results
        with open(full_path, "wb") as fp:
            pickle.dump(init, fp)
        return init

def get_pretrained_glove_vectors(path=glove_embeddings_path, load_files=False):
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
        if embedding_size == 100:
            glove_vectors = open(os.path.join(glove_directory, "glove.6B.100d.txt"))
        else:
            glove_vectors = open(os.path.join(glove_directory, "glove.6B.50d.txt"))
        for line in glove_vectors:
            values = line.split()
            word = values[0]
            coefficients = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefficients
        glove_vectors.close()
        with open(path, "wb") as fp:
            pickle.dump(embeddings_index, fp)
        return embeddings_index

def get_sentence_from_tokens(tokenized_sentence, word_index_reverse):
    """ Reconstructs sentence from list of tokens.

    Args:
        tokenized_sentence: list of tokens, where each token < vocab_size
        word_index_reverse: a dictionary of shape {word_index: word}
    Returns:
        sentence: the reconstructed sentence
    """
    sentence = [word_index_reverse[word] for word in tokenized_sentence]
    print 'tokens:    {}'.format(tokenized_sentence)
    print 'sentence:  {}'.format(sentence)
    return sentence

def print_tokenizer_information(tokenizer=None, corpus=None):
    """ Prints useful information about the tokenizer.

    Args:
        tokenizer: word tokenizer
        corpus: list of sentences
    """
    if tokenizer is None:
        corpus = pd.read_csv('train.csv').fillna(' ')[["comment_text"]].values.flatten()
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(corpus)
    print 'word_index'
    print tokenizer.word_index
    print 'word_counts (frequency)'
    print tokenizer.word_counts
    print 'corpus'
    print corpus
    print 'sequences'
    print tokenizer.texts_to_sequences(corpus)
    print 'vocab size'
    print len(tokenizer.word_index.keys())

def test_build_coccurrence_matrix():
    """ Tests the model with a small arbitrary data set
    """
    corpus = get_development_data()
    cooccur, tokenizer = build_coccurrence_matrix(corpus, min_frequency=2)
    print_tokenizer_information(tokenizer, corpus)
    print cooccur

def test_train():
    """ Tests the cooccurrence matrix with a small dataset
    """
    # Build cooccurrence matrix
    corpus = get_development_data()
    cooccurrence_matrix, tokenizer  = build_coccurrence_matrix(corpus, min_frequency=2)
    vocab_size = len(tokenizer.word_index.keys())
    embeddings = build_graph_and_train(cooccurrence_matrix, vocab_size, "dev_test", tokenizer)
    print "Final embeddings:"
    print embeddings[1]

def test_glove_model(scope):
    """Tests the model using the first fifteen elements in the training data sets

    Args:
        scope: variable name scope for the graph
    """
    corpus = get_development_data()
    cooccurrence_matrix, tokenizer  = build_coccurrence_matrix(corpus)
    vocab_size = len(tokenizer.word_index.keys())
    embeddings = build_graph_and_train(cooccurrence_matrix, vocab_size, scope, tokenizer)
    print "Final embeddings shape {}:".format(np.array(embeddings).shape)
    print embeddings[0]

def test_minibatch():
    """ Tests minibatch using small data set.
    """
    corpus = get_development_data()
    cooccurrence_matrix, tokenizer  = build_coccurrence_matrix(corpus)
    minibatches = get_cooccurrence_batches(cooccurrence_matrix, 5)
    for batch in minibatches:
        i, j, X_ij = batch
        print 'i:       {}'.format(i)
        print 'j:       {}'.format(j)
        print 'count:   {}'.format(X_ij)

def test_f():
    """ Tests the function for preventing common word pairs
    """
    x_ij_max = 100
    alpha = 0.75
    corpus = get_development_data()
    cooccurrence_matrix, tokenizer  = build_coccurrence_matrix(corpus)
    minibatches = get_cooccurrence_batches(cooccurrence_matrix, 5)
    for batch in minibatches:
        i, j, X_ij = batch
        print 'count batch: {}'.format(X_ij)
        f = tf.map_fn(lambda x_ij: tf.cond(x_ij < x_ij_max, lambda: tf.pow(tf.divide(x_ij, x_ij_max), alpha), lambda: tf.cast(1.0, tf.float64)), X_ij)
        with tf.Session() as sess:
            print'f: {}'.format(sess.run(f))
            sess.close()
        # just need to check for one batch
        return

def test_gpu():
    print "Device: {}".format(device)
    with tf.device(device):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(sess.run(c))
    sess.close()

if __name__ == "__main__":
    myargs = getopts(argv)
    print myargs
    if "-bs" in myargs:
        batch_size = int(myargs["-bs"])

    if "-mn" in myargs:
        if myargs["-mn"] == "gpu":
            device = "/gpu:0"

    if "-ep" in myargs:
        num_epochs = int(myargs["-ep"])

    if "-em" in myargs:
        embedding_size = int(myargs["-em"])

    if "-nm" in myargs:
        num_words = int(myargs["-nm"])

    if "-pt" in myargs:
        initialize_with_pretrained = bool(myargs["-pt"])

    if "-wn" in myargs:
        window_size = int(myargs["-wn"])

    if "-lr" in myargs:
        learning_rate = int(myargs["-lr"])

    if "-run" in myargs:
        run_arg = myargs["-run"]
        if run_arg == "all":
            generate_embeddings_all()

    if "-test" in myargs:
        if myargs["-test"] == "glove":
            test_glove_model("development")
        elif myargs["-test"] == "minibatch":
            test_minibatch()
        elif myargs["-test"] == "train":
            test_train()
        elif myargs["-test"] == "cooccur":
            test_build_coccurrence_matrix()
        elif myargs["-test"] == "gpu":
            device = "/gpu:0"
            test_gpu()
        elif myargs["-test"] == "f":
            test_f()
