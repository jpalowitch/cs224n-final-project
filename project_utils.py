import numpy as np
import tensorflow as tf

kColumnNames = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', \
                'identity_hate']

kTrainSplitSeed = 12345


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

def get_train_dev_split(df, train_prop, seed=kTrainSplitSeed):
  """Takes pd.DataFrame from load of data and gives a train/dev split.
  
  Args:
    data: a pd.DataFrame of the jigsaw data.
    train_prop: proportion of data you want for the training set.
    seed: an integer random seed for the split.
  Returns:
    train: training data.
    dev: testing data.
  """
  ntrain = int(df.shape[0] * train_prop)
  df = df.sample(frac=1)
  return df[:ntrain], df[ntrain - df.shape[0]:]

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
      
