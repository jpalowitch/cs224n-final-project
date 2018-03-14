import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import urllib
from nltk import word_tokenize
from string import punctuation
PUNCTUATION = [punctuation[i:i+1] for i in range(0, len(punctuation), 1)]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
text = tf.keras.preprocessing.text
sequence = tf.keras.preprocessing.sequence

# Constants
CLASS_NAMES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', \
               'identity_hate']
SPLIT_SEED = 123454321
RUN_SEED = 543212345
SPLIT_PROP = [3.0, 1.0, 1.0]
TFIDF_VECTORS_FILE_TOXIC = "tfidf_sentence_vectors.pkl"
TFIDF_VECTORS_FILE_AGG = "tfidf_sentence_vectors_aggresion.pkl"
TRAIN_DATA_FILE = "train.csv"
NUM_FEATURES = 10000
SESS_SAVE_DIRECTORY = "sess_saves"
ATTACK_AGGRESSION_FN = "attack_aggression_data.csv"
EMBEDDING_FILE = 'glove.6B/glove.6B.100d.txt' # Originally 300d

def get_base2_labels(rows):
    """Converts a matrix of binary row vectors to labels.

    Args:
      rows: an array of binary vectors.
    Returns:
      labels: a column vector of integers = base 2 versions of rows.
    """
    base2_vec = [2 ** x for x in range(rows.shape[1])]
    return np.matmul(rows, base2_vec)

def get_base2_onehots(rows):
    """Converts a matrix of binary row vectors to one-hot label vectors.

    Args:
      rows: an array of binary vectors.
    Returns:
      labels: a row matrix of one-hots giving base2 classes of rows.
    """
    base2_mat = np.zeros((rows.shape[0], pow(2, rows.shape[1])))
    base2_list = list(get_base2_labels(rows))
    one_vec = np.ones((rows.shape[0],))
    base2_mat[range(rows.shape[0]), get_base2_labels(rows)] = one_vec
    return base2_mat

def get_onehots_from_labels(labels):
    """Converts an np integer vector of labels into a matrix of one-hot vectors.

    Args:
      labels: an integer vector of labels:
    Returns:
      onehots: a row matrix of one-hots. Each row as a 1 in position i if i is the
      position of the row's integer in the ordered integer labels.
    """
    unique_labs = list(set(labels))
    label_indx = [unique_labs.index(x) for x in labels]
    one_vec = np.ones((labels.shape[0], ))
    onehots = np.zeros((labels.shape[0], len(unique_labs)))
    onehots[range(labels.shape[0]), label_indx] = one_vec
    return onehots

def get_TDT_split(df, split_prop=SPLIT_PROP, seed=SPLIT_SEED):
    """Takes pd.DataFrame from load of data and gives a train/dev split.

    Args:
      data: a pd.DataFrame of the jigsaw data.
      split_prop: a list of floats which is proportional to data split.
      seed: an integer random seed for the split.
    Returns:
      train: training data.
      dev: development data.
      test: testing data.
    """
    ndata = [int(df.shape[0] * x / sum(split_prop)) for x in split_prop]
    ndata[2] = ndata[2] + df.shape[0] - sum(ndata)
    np.random.seed(seed)
    df = df.sample(frac=1)
    train = df[:ndata[0]]
    dev = df[ndata[0]:(ndata[0] + ndata[1])]
    test = df[-ndata[2]:]
    return train, dev, test

def test_get_TDT_split():
    data = pd.read_csv('train.csv').fillna(' ')
    d11, d12, d13 = get_TDT_split(data)
    d21, d22, d23 = get_TDT_split(data)
    d31, d32, d33 = get_TDT_split(data)
    assert d11[['comment_text']].values.flatten()[99] == \
           d21[['comment_text']].values.flatten()[99] == \
           d31[['comment_text']].values.flatten()[99]
    print 'Random seed returned same values'
    print  d31[['comment_text']].values.flatten()[99]

def get_development_data():
    """Reads csv data and returns a very small portion of it for building models
    """
    data, _, _ = get_TDT_split(pd.read_csv('train.csv').fillna(' '))
    return data[["comment_text"]].values.flatten()[:15]


def get_sparse_input(scipy_sparse):
    """Produces needed input to tf.sparse_placeholder from a csr matrix

    Args:
      scipy_sparse: a scipy sparse CSR matrix
    Returns:
      sparse_input: a tuple to pass to sparse_placeholder, see examples at:
        https://www.tensorflow.org/versions/r0.12/api_docs/python/io_ops/placeholders
    """
    coo = scipy_sparse.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    sparse_input = (indices, coo.data, coo.shape)
    return sparse_input

def calc_auc(labels, probs, mean=True):
    """Takes an array of *individual probabilities* and a comparison array of
       binary indiicators and computes the average ROC AUC for the entire array.

    Args:
       labels: a binary indicator array
       probs: an array with each entry between 0 and 1 (unrestricted by other
           entries)
       mean: if True, returns mean of column-wise AUC. if False, returns aucs
           across columns.
    Returns:
       scalar, average ROC-AUC of probs
    """
    aucs = []
    if len(probs.shape) > 1:
        for i in range(probs.shape[1]):
            aucs.append(roc_auc_score(labels[:, i], probs[:, i]))
    else:
        aucs.append(roc_auc_score(labels, probs))
    if mean:
        return np.mean(aucs)
    else:
        return aucs


def save_auc_scores(scores, approach, classifier, flavor,
                    fn="auc_scores.csv", overwrite=True, cnames=CLASS_NAMES):
    """Records auc scores of approach-flavor run.

	   ***Before setting your approach/classifier/flavor strings, make sure to
   	      check out the existing auc_scores.csv for formatting. This will help
		  later to visualize results from particular approaches/classifiers.***

    Args:
      scores: a list or array of 6 auc scores
      approach: string that names the approach
      flavor: string that names the flavor
      fn: output filename
      overwrite: if True, will overwrite a previous result with the same
        approach & flavor
    Returns:
      None
    """
    new_data_d = {"Approach": approach,
                  "Classifier": classifier,
                  "Flavor": flavor}
    new_data_d.update(zip(cnames, scores))
    if os.path.isfile(fn):
        old_data = pd.read_csv(fn, index_col=0)
        new_data = pd.DataFrame(data=new_data_d, index=[old_data.shape[0]])
        old_data = old_data.append(new_data)
        if overwrite:
            old_data = old_data.drop_duplicates(
                subset=['Approach', 'Classifier', 'Flavor'], keep='last')
        else:
            old_data = old_data.drop_duplicates(
                subset=['Approach', 'Classifier', 'Flavor'], keep='first')
    else:
        old_data = pd.DataFrame(data=new_data_d, index=[0])
    old_data.to_csv(fn)
    return None

def save_rnn_auc_scores(scores, fields, dataset, cnames, overwrite=True):
    """Records auc scores of rnn_tensorflow.py runs
    """
    fn = "auc_scores_rnn_" + dataset + ".csv"
    ow_fields = list(fields.keys())
    fields.update(zip(cnames, scores))
    if os.path.isfile(fn):
        old_data = pd.read_csv(fn, index_col=0)
        new_data = pd.DataFrame(data=fields, index=[old_data.shape[0]])
        old_data = old_data.append(new_data)
        if overwrite:
            old_data = old_data.drop_duplicates(subset=ow_fields, keep='last')
        else:
            old_data = old_data.drop_duplicates(subset=ow_fields, keep='first')
    else:
        old_data = pd.DataFrame(data=fields, index=[0])
    old_data.to_csv(fn)
    return None

def vectorize_corpus_tf_idf(train, dev, test, path=TFIDF_VECTORS_FILE_TOXIC,
                            n_features=NUM_FEATURES, sparse=False):
    """ Vectorizes the corpus using tf-idf. Saves in sparse format. Also saves
        the vectorizer object for potential later use on new examples.

    Args:
        train: train split of kaggle-formatted data
        dev: dev split of kaggle-formatted data
        test: test split of kaggle-formatted data
        path: path to data file
        n_features: max number of ngram features to count
        sparse: if True, returns feature vecs in original sparse format. Else,
            they are returned as numpy arrays
    Returns:
        train_vecs: tfidf vectors for training data
        dev_vecs: tfidf vectors for dev data
        test_vecs: tfidf vectors for test data
    """
    # Computing and saving
    if os.path.isfile(path):
        print("Using stored word vectors.")
        with open(path, "rb") as fp:
            sentence_vectors = pickle.load(fp)
    else:
        print("Word vector file path not found. Computing word vectors.")
        vectorizer = TfidfVectorizer(
            max_features=n_features,
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(1, 2)
        )
        train_text = train['comment_text']
        dev_text = dev['comment_text']
        test_text = test['comment_text']
        vectorizer.fit(pd.concat([train_text, dev_text, test_text]))
        sentence_vectors = {
            'train_vecs': hstack([vectorizer.transform(train_text)]),
            'dev_vecs': hstack([vectorizer.transform(dev_text)]),
            'test_vecs': hstack([vectorizer.transform(test_text)]),
            'vectorizer': vectorizer}
        with open(path, "wb") as fp:
            pickle.dump(sentence_vectors, fp)

    # Extracting and returning
    train_vecs = sentence_vectors['train_vecs']
    dev_vecs = sentence_vectors['dev_vecs']
    test_vecs = sentence_vectors['test_vecs']
    if not sparse:
        train_vecs = train_vecs.toarray()
        dev_vecs = dev_vecs.toarray()
        test_vecs = test_vecs.toarray()
    return train_vecs, dev_vecs, test_vecs

def get_stock_embeddings(X_train, X_dev, X_test, embed_file=EMBEDDING_FILE, embed_size=100, max_features=10000):
    """ Gets stock embeddings. Adapted from
    https://www.kaggle.com/prashantkikani/pooled-gru-glove-with-preprocessing
    """
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train) + list(X_dev) + list(X_test))
    X_train = tokenizer.texts_to_sequences(X_train)
    X_dev = tokenizer.texts_to_sequences(X_dev)
    X_test = tokenizer.texts_to_sequences(X_test)

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embed_file))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return X_train, X_dev, X_test, embedding_matrix

def minibatch(inputs, labels, batch_size, shuffle=True, masks=None):
    """ Performs minibatching on set of data. Based off of stack overflow post:
    https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python

    Args:
        inputs: feature matrix
        labels: label vector
        batch_size: size of batch to sample
        shuffle: whether to randomly shuffle indices
    Returns:
        a batch of inputs and labels
    """
    assert inputs.shape[0] == labels.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for i in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        if shuffle:
            batch = indices[i:(i + batch_size)]
        else:
            batch = slice(i, i + batch_size)
        if masks is not None:
            yield inputs[batch], labels[batch], masks[batch]
        else:
            yield inputs[batch], labels[batch]

def saver_fn(approach, classifier, flavor, class_name='all'):
    return './%s/%s_%s_%s_class=%s.weights' % (SESS_SAVE_DIRECTORY, \
        approach, classifier, flavor, class_name)

def saver_fn_rnn(fields, class_name='all'):
    fkeys = [str(x) for x in list(fields.keys())]
    fvals = [str(x) for x in list(fields.values())]
    ids = [a + '-' + b for a, b in zip(fkeys, fvals)]
    fn = './sess_saves/' + '_'.join(ids) + '_' + class_name + '.weights'
    return fn

def getopts(argv):
    """ Gets and parses command-line arguments.

    Args:
        inputs: the argument input object
    Returns:
        the parsed arguments
    """
    opts = {}
    while argv:
        if argv[0][0] == '-':
            opts[argv[0]] = argv[1]
        argv = argv[1:]
    return opts

def tokenize(comment):
	'''
	for one comment, tokenizes, removes punctuation and changes to lowercase
	'''
	words = word_tokenize(comment)
	words = [w.lower() for w in words]
	words = [w for w in words if w not in PUNCTUATION and not w.isdigit()]
	return words


def preprocess_seqs(inputs, max_length=None, method=None):
    """ Takes indexed sentences and prepares the data for RNN input.

    Args:
        inputs: list of index lists as returned by get_word_embeddings().
        method: string which is either random or truncate, if random, uses downsampling,
        if truncate, cuts off tokens after max_length
    Returns:
        inputs_mat: a row-mat of index lists that have been padded or shortened.
        masks: a row-mat of max_length-length boolean masks for each sentence.
    """
    new_inputs = []
    masks = []

    for sentence in inputs:
        T = len(sentence)
        if T > max_length:
            if method == 'random':
                sentence2 = np.random.choice(
                    sentence, size=max_length, replace=False
                )
                mask = [True] * max_length
            else:
                sentence2 = sentence[:max_length]
                mask = [True] * max_length
        else:
            sentence2 = sentence + [0] * (max_length - T)
            mask = [True] * T + [False] * (max_length - T)
        new_inputs.append(sentence2)
        masks.append(mask)
    inputs_mat = np.array(new_inputs).astype(np.int32)
    return inputs_mat, np.array(masks)

def get_and_save_talk_data():
    """ Function to download, save, and pre-process wikipedia talk share data:
    https://figshare.com/projects/Wikipedia_Talk/16731
    """

    # Downloading
    if not os.path.isfile('personal_attack_comments.tsv'):
        print("downloading personal_attack_comments")
        urllib.urlretrieve('https://ndownloader.figshare.com/files/7554634',
                           'personal_attack_comments.tsv')
        print("--done")
    if not os.path.isfile('personal_attack_annotations.tsv'):
        print("downloading personal_attack_annotations")
        urllib.urlretrieve('https://ndownloader.figshare.com/files/7554637',
                           'personal_attack_annotations.tsv')
        print("--done")
    if not os.path.isfile('aggression_comments.tsv'):
        print("downloading aggression_comments")
        urllib.urlretrieve('https://ndownloader.figshare.com/files/7038038',
                           'aggression_comments.tsv')
        print("--done")
    if not os.path.isfile('aggression_annotations.tsv'):
        print("downloading aggression_attack_comments")
        urllib.urlretrieve('https://ndownloader.figshare.com/files/7394506',
                           'aggression_annotations.tsv')
        print("--done")

    # Pre-processing
    def get_csv(fn):
        return pd.read_csv(fn, sep='\t', index_col=0)
    att_com = get_csv('personal_attack_comments.tsv')
    att_ann = get_csv('personal_attack_annotations.tsv')
    agg_com = get_csv('aggression_comments.tsv')
    agg_ann = get_csv('aggression_annotations.tsv')
    att_labels = att_ann.groupby('rev_id')['attack'].mean() > 0.5
    agg_labels = agg_ann.groupby('rev_id')['aggression'].mean() > 0.5
    att_labels = att_labels.astype('int32')
    agg_labels = agg_labels.astype('int32')
    att_com['comment'] = att_com['comment'].apply(
        lambda x: x.replace("NEWLINE_TOKEN", " "))
    att_com['comment'] = att_com['comment'].apply(
        lambda x: x.replace("TAB_TOKEN", " "))
    comments_d = {'comment_text': att_com['comment'].values,
                  'attack': att_labels.values,
                  'aggression': agg_labels.values}
    comments_df = pd.DataFrame(comments_d)
    comments_df.to_csv(ATTACK_AGGRESSION_FN)
