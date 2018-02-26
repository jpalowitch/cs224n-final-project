import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Constants
CLASS_NAMES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', \
               'identity_hate']
SPLIT_SEED = 123454321
RUN_SEED = 543212345
SPLIT_PROP = [3.0, 1.0, 1.0]
TFIDF_VECTORS_FILE = "tfidf_sentence_vectors.pkl"
TRAIN_DATA_FILE = "train.csv"
NUM_FEATURES = 10000

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

def sparse_mat_to_sparse_tensor(scipy_sparse):
  """Converts a sparse matrix to a tensor object.

  Args:
    scipy_sparse: a scipy sparse CSR matrix
  Returns:
    tf_sparse: a tensorflow sparse matrix
  """
  coo = scipy_sparse.tocoo()
  indices = np.mat([coo.row, coo.col]).transpose()
  tf_sparse = tf.SparseTensor(indices, coo.data, coo.shape)
  return tf_sparse

def save_auc_scores(scores, approach, flavor, 
                    fn="auc_scores.csv", overwrite=True):
  """Records auc scores of approach-flavor run.
  
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
  new_data_d = {"Approach": approach, "Flavor": flavor}
  new_data_d.update(zip(CLASS_NAMES, scores))
  if os.path.isfile(fn):
    old_data = pd.read_csv(fn, index_col=0)
    new_data = pd.DataFrame(data=new_data_d, index=[old_data.shape[0]])
    old_data = old_data.append(new_data)
    if overwrite:
      old_data = old_data.drop_duplicates(subset=['Approach', 'Flavor'])
  else:
    old_data = pd.DataFrame(data=new_data_d, index=[0])
  old_data.to_csv(fn)
  return None
    
def vectorize_corpus_tf_idf(train, dev, test, path=TFIDF_VECTORS_FILE,
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
