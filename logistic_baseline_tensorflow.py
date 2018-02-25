'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
from project_utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from logistic_baseline_sklearn import get_features
from scipy.sparse import csr
from sklearn.metrics import roc_auc_score

import pandas as pd

# Parameters
learning_rate = 0.01
training_epochs = 50
batch_size = 1000
display_step = 1
n_features = 10000
target_class = 0

# Get data and featurize
train, dev, test = get_TDT_split(pd.read_csv('train.csv').fillna(' '))
train_text = train['comment_text']
dev_text = dev['comment_text']
test_text = test['comment_text']
train_features, dev_features = get_features(
  train_text=train['comment_text'],
  dev_text=dev['comment_text'],
  vocab=pd.concat([train_text, dev_text, test_text]),
  sparse=False,
  max_features=n_features)
train_target = get_onehots_from_labels(train[kClassNames[target_class]].values)
dev_labels = dev[kClassNames[target_class]].values
dev_target = get_onehots_from_labels(dev_labels)
num_examples = train_features.shape[0]
n_classes = train_target.shape[1]

# tf Graph Input
x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.float32, [None, n_classes])

# Set model weights
W = tf.get_variable(
      "weights",
      shape=[n_features, n_classes],
      initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.zeros([n_classes]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# Gradient Descent
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Final scoring
def AUC_calc(X, Y): 
  return roc_auc_score(Y[:, 1], pred.eval({x: X})[:, 1])

# Initialize the variables (i.e. assign their default value)
global_init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
  
  # Run initializer
  sess.run(global_init)
  
  # Training 
  for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(num_examples/batch_size)
    # Loop over batches
    for i in range(total_batch):
      upper_indx = min((i + 1) * batch_size, num_examples)
      i_indxs = range(i * batch_size, upper_indx)
      batch_xs = train_features[i_indxs]
      batch_ys = train_target[i_indxs]
      _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                    y: batch_ys})
      avg_cost += c / total_batch
    # Display logs
    if (epoch+1) % display_step == 0:
      print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
  
  print("Optimization Finished!")
  print ("AUC_ROC:", AUC_calc(dev_features, dev_target))
  
  sess.close()

