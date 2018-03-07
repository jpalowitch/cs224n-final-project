from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from project_utils import tokenize as word_tokenizer
from scipy import sparse
import numpy as np
from collections import defaultdict


def build_coccurrence_matrix(corpus, window_size=10, min_frequency=0):
    """ Builds a cooccurence matrix as a dictionary.

    Args:
        corpus: list of sentences
        window_size: size of the window to consider word frequencies in
        min_frequency: minimum frequency of a word to keep

    Returns:
        cooccurence_matrix: dictionary of form {(center_word, context_word):count}
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
        # tokenized_sentence = word_tokenizer(sentence.decode("utf-8"))
        print 'sequence: {}'.format(token_ids)
        # v represents the center word; u is the context vector
        for v_idx, v in enumerate(token_ids):
            # separate contexts for left side to use for both, because symmetry
            left_context_words = token_ids[max(0, v_idx - window_size) : v_idx]
            context_size = len(left_context_words)

            # TODO: figure out whether to break this into lhs and rhs
            # left context
            for u_idx, u in enumerate(left_context_words):
                distance = context_size - u_idx
                # weight by distance
                cooccurence_matrix_unfiltered[(u, v)] += (1.0 / float(distance))
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

        return cooccurence_matrix
    else:
        return cooccurence_matrix_unfiltered

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
    corpus = ['San Francisco is in California.', 'California is a great place.', 'California is a  subpar place.']
    cooccur = build_coccurrence_matrix(corpus, min_frequency=2)
    print cooccur

if __name__ == '__main__':
    test_build_coccurrence_matrix()
