import numpy as np


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
    rows: an array of one-hot rows, some may not have ones
  Returns:
    labels: a column vector of labels, one for each row.
  """
  return np.apply_along_axis(get_label, 1, rows)
