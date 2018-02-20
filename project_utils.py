import numpy as np
import pandas as pd

kColumnNames = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', \
                'identity_hate']


def get_label(row):
  """Converts a one-hot vector to a label.
  
  Args:
    row: a one-hot vector, potentially without a one.
  Returns:
    label: index position of the one in row. returns zero if there isn't a one.
  """
  zero_index = np.nonzero(row)
  if zero_index[0].shape[0] == 0:
    label = 0
  elif zero_index[0].shape[0] == 1:
    label = zero_index[0][0] + 1
  return label


def get_labels(rows):
  """Converts an array of one-hot rows to labels with get_label.
  
  Args:
    rows: an array of one-hot rows, some may not have ones.
  Returns:
    labels: a column vector of labels, one for each row.
  """
  return np.apply_along_axis(get_label, 1, rows)

def make_labels_from_df(df):
  """Converts jigsaw-formatted pandas df into vector of labels.
  
  Args:
    df: a pandas.DataFrame of any subset of the jigsaw data.
  Returns:
    labels: a column vector of labels corresponding to the rows of df.
  """
  rows = df.as_matrix(kColumnNames)
  return get_labels(rows)
