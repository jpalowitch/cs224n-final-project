import numpy as np

kColumnNames = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', \
                'identity_hate']


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
      
