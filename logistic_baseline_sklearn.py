import numpy as np
import pandas as pd

from project_utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack

  
def get_features(train_text, dev_text, vocab):
  """Gets feature mats for train and test
  
  Args:
    train_text: pd.Series of test comments
    dev_text: pd.Series of dev comments
    vocab: a word vocabulary on which to featurize
  Returns:
    train_features: a sparse matrix of word ngram features from train
    dev_features: a sparse matrix of word ngram features from dev
  """

  # Getting word vectorizer
  print('word featurizing...')
  word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=10000)

  # Getting features
  word_vectorizer.fit(vocab)
  train_word_features = word_vectorizer.transform(train_text)
  dev_word_features = word_vectorizer.transform(dev_text)
  train_features = hstack([train_word_features])
  dev_features = hstack([dev_word_features])
  
  return train_features, dev_features


def fit_model(train, dev):
  """Fits and evals logistic regression model.
  
  Args:
    train: a pd.DataFrame of the training data.
    dev: a pd.DataFrame of the dev data.
  Returns:
    average ROC-AUC score over classes.
  """
  
  # Getting features
  train_text = train['comment_text']
  dev_text = dev['comment_text']
  test_text = test['comment_text']
  train_features, dev_features = get_features(
    train_text=train['comment_text'],
    dev_text=dev['comment_text'],
    vocab=pd.concat([train_text, dev_text, test_text]))
  
  # Doing one-vs-all training
  scores = []
  for class_name in kClassNames:
    print('doing class {}'.format(class_name))
    train_target = train[class_name]
    dev_target = get_onehots_from_labels(dev[class_name].values)
    classifier = LogisticRegression(solver='sag')
    model = classifier.fit(train_features, train_target)
    dev_pred = model.predict_proba(dev_features)
    ROC_AUC_score = roc_auc_score(dev_target, dev_pred)
    scores.append(ROC_AUC_score)
    print('--CV score is {}'.format(ROC_AUC_score))

  return np.mean(scores)


if __name__ == '__main__':
  train, dev, test = get_TDT_split(pd.read_csv('train.csv').fillna(' '))
  avg_ROC_AUC = fit_model(train, dev)
  print('Total CV score is {}'.format(avg_ROC_AUC))
