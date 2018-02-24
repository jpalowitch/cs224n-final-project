import numpy as np
import tensorflow as tf

# Constants
kClassNames = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', \
               'identity_hate']
kSplitSeed = 123454321
kSplitProp = [3.0, 1.0, 1.0]


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

def get_TDT_split(df, split_prop=kSplitProp, seed=kSplitSeed):
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
  df = df.sample(frac=1)
  train = df[:ndata[0]]
  dev = df[ndata[0]:(ndata[0] + ndata[1])]
  test = df[ndata[2] - df.shape[0]:]
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
      
