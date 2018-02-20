import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

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
