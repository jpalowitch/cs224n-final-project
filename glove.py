from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from project_utils import tokenize as word_tokenizer
from scipy import sparse
import numpy as np
from collections import defaultdict
import tensorflow as tf


def build_coccurrence_matrix(corpus, window_size=10, min_frequency=0):
    """ Builds a cooccurence matrix as a dictionary.

    Args:
        corpus: list of sentences
        window_size: size of the window to consider word frequencies in
        min_frequency: minimum frequency of a word to keep

    Returns:
        cooccurence_matrix: dictionary of form {(center_word, context_word):count}
        tokenizer: word tokenizer to fetch information for word frequency and other values
    """
    print 'Building cooccurence matrix'
    # train tokenizer on corpus
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    # print_tokenizer_information(tokenizer, corpus)

    # create dict of {token_index: word}
    word_index_reverse = {v:k for k, v in tokenizer.word_index.items()}

    cooccurence_matrix_unfiltered = defaultdict(float)
    sequences = tokenizer.texts_to_sequences(corpus)

    for idx, token_ids in enumerate(sequences):
        if idx % 100 == 0:
            print 'On line: {}'.format(idx)
        # uncomment to use word tokenizer instead of default one
        # print 'sequence: {}'.format(token_ids)
        # v represents the center word; u is the context vector
        for v_idx, v in enumerate(token_ids):
            # separate contexts for left side to use for both, because symmetry
            left_context_words = token_ids[max(0, v_idx - window_size) : v_idx]
            right_context_words = token_ids[(v_idx + 1) : min(len(token_ids), v_idx + window_size + 1)]
            left_context_size = len(left_context_words)
            right_context_size = len(right_context_words)

            # left context word
            for u_idx, u in enumerate(left_context_words):
                distance = left_context_size - u_idx
                cooccurence_matrix_unfiltered[(v, u)] += (1.0 / float(distance))

            # right context words
            for u_idx, u in enumerate(right_context_words):
                distance = u_idx + 1
                cooccurence_matrix_unfiltered[(v, u)] += (1.0 / float(distance))

    if min_frequency > 1:
        # matrix of (v, u) : count
        cooccurence_matrix = defaultdict(float)
        for (v, u), count in cooccurence_matrix_unfiltered.items():
            v_actual = word_index_reverse[v]
            u_actual = word_index_reverse[u]
            # print 'v: {} u: {} count: {}'.format(v, u, count)
            if tokenizer.word_counts[v_actual] >= min_frequency and tokenizer.word_counts[u_actual] >= min_frequency:
                 cooccurence_matrix[(v, u)] = count

        return cooccurence_matrix, tokenizer
    else:
        return cooccurence_matrix_unfiltered, tokenizer


def build_graph_and_train(cooccurence_matrix, vocab_size):
    """ Builds and trains a tensorflow model for creating GloVe vectors based off
        of a cooccurence_matrix.

    Args:
        cooccurence_matrix: dictionary of form {(input_word_index, context_word_index) : count}
        vocab_size: size of vocabulary

    Returns:
        embeddings: (vocab_size x embedding_size) shape matrix of word vectors
    """
    # model params
    embedding_size = 50
    batch_size = 1
    alpha = 0.75
    num_epochs = 2
    learning_rate = 0.2

    # upper bound for words that cooccur frequently
    x_ij_max = 100.0

    # variables
    # weights for input (i) and context (j) vectors
    W_i = tf.get_variable("W_i", shape=(vocab_size, embedding_size), initializer=tf.contrib.layers.xavier_initializer())
    W_j = tf.get_variable("W_j", shape=(vocab_size, embedding_size), initializer=tf.contrib.layers.xavier_initializer())

    # each word has a scalar weight for when it is a center or context word
    B_i = tf.get_variable("input_bias", shape=(vocab_size), initializer=tf.constant_initializer)
    B_j = tf.get_variable("context_bias", shape=(vocab_size), initializer=tf.constant_initializer)

    # add placeholders for indices for the input and output words and cooccurence for (v, u)
    i = tf.placeholder(tf.int32, shape=(batch_size), name="input_batch")
    j = tf.placeholder(tf.int32, shape=(batch_size), name="output_batch")
    X_ij = tf.placeholder(tf.float32, shape=(batch_size), name="context_batch")

    # weights and biases
    w_i = tf.nn.embedding_lookup([W_i], i)
    w_j = tf.nn.embedding_lookup([W_j], j)
    b_i = tf.nn.embedding_lookup([B_i], i)
    b_j = tf.nn.embedding_lookup([B_j], j)

    # actual math
    # f (X_ij): map the function count < max_count ? (count/max_count)^2 : 1 onto each element
    f = tf.map_fn(lambda x_ij: tf.cond(x_ij < x_ij_max, lambda: tf.square(tf.divide(x_ij, x_ij_max)), lambda: 1.0), X_ij)

    # w_i * w_j + b_i + b_j + log(X_ij)
    # the weights are row vectors so need to calculate outer product
    w_j_transpose = tf.transpose(w_j)
    inner_value = tf.matmul(w_i, w_j_transpose) + b_i + b_j - tf.log(X_ij)

    # loss: f(X_ij) * [(w_i * w_j + b_i + b_j + log(X_ij))^2]
    f_expaneded = tf.expand_dims(f, 1)
    loss = tf.matmul(f_expaneded, tf.pow(inner_value, 2))

    # reduce for entire batch
    total_loss = tf.reduce_sum(loss)

    # set optimizer
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

    # treating final word vectors as i + j
    combined_embeddings = W_i + W_j

    # train!!
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(num_epochs):
            # TODO: add minibatching, remove array syntax
            for (input_batch,context_batch), count_batch in cooccurence_matrix.items():
                # make sure everything is the right shape
                # f_array = sess.run(f, feed_dict={X_ij: [200, 20]})
                # x_array = sess.run(X_ij, feed_dict={X_ij: [200, 20]})
                # print 'f_array: {}'.format(f_array)
                # print 'x_array: {}'.format(x_array)
                # assert f_array.shape == x_array.shape

                # finally train
                feed_dict = {
                    i: [input_batch],
                    j: [context_batch],
                    X_ij: [count_batch]
                }
                sess.run(optimizer, feed_dict=feed_dict)

        embeddings = sess.run(combined_embeddings)
        return embeddings


###### UTILS
def get_sentence_from_tokens(tokenized_sentence, word_index_reverse):
    sentence = [word_index_reverse[word] for word in tokenized_sentence]
    print 'tokens:    {}'.format(tokenized_sentence)
    print 'sentence:  {}'.format(sentence)
    return sentence

def print_tokenizer_information(tokenizer, corpus):
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
    corpus = ['San Francisco is in California.', 'California is a great place.', 'California is a subpar place.']
    cooccur = build_coccurrence_matrix(corpus, min_frequency=2)
    print cooccur

def test_train():
    # Build cooccurence matrix
    corpus = ['San Francisco is in California.', 'California is a great place.', 'California is a subpar place.']
    cooccurrence_matrix, tokenizer  = build_coccurrence_matrix(corpus, min_frequency=2)
    vocab_size = len(tokenizer.word_index.keys())
    embeddings = build_graph_and_train(cooccurrence_matrix, vocab_size)
    print 'final embeddings:'
    print embeddings    

if __name__ == '__main__':
    test_train()
