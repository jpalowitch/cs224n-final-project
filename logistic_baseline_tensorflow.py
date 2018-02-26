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
from scipy.sparse import csr
from sklearn.metrics import roc_auc_score

import pandas as pd

APPROACH = "ngram"
CLASSIFIER = "logistic"
FLAVOR = "tensorflow-ADAM"

# Parameters
learning_rate = 0.01
training_epochs = 50
batch_size = 1000
display_step = 1

# Get data and featurizing
train, dev, test = get_TDT_split(pd.read_csv('train.csv').fillna(' '))
#train_vecs, dev_vecs, test_vecs = vectorize_corpus_tf_idf(train, dev, test)
train_vecs, dev_vecs, test_vecs = vectorize_corpus_tf_idf(train, dev, test, sparse=True)
n_train = train_vecs.shape[0]

# tf Graph Input
#x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
x = tf.sparse_placeholder(tf.float32)
y = tf.placeholder(tf.float32, [None, 2])

# Set model weights
W = tf.get_variable("weights",
                    shape=[NUM_FEATURES, 2],
                    initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.zeros([2]))

# Construct model
#pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
pred = tf.nn.softmax(tf.sparse_tensor_dense_matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# Gradient Descent
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Final scoring
def AUC_calc(X, Y): 
    return roc_auc_score(Y[:, 1], pred.eval({x: X})[:, 1])

# Initialize the variables (i.e. assign their default value)
global_init = tf.global_variables_initializer()

auc_scores = []
for target_class in range(1):
    
    # Getting labels for training
    train_labels = train[CLASS_NAMES[target_class]].values
    train_target = get_onehots_from_labels(train_labels)
    test_labels = test[CLASS_NAMES[target_class]].values
    test_target = get_onehots_from_labels(test_labels)
    
    # Start training
    with tf.Session() as sess:
    
        # Run initializer
        sess.run(global_init)
        
        # Training 
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_train/batch_size)
            
            # Loop over batches
            for i in range(total_batch):
                upper_indx = min((i + 1) * batch_size, n_train)
                i_indxs = range(i * batch_size, upper_indx)
                batch_xs = sparse_tens_from_sparse_mat(train_vecs[i_indxs])
                batch_ys = train_target[i_indxs]
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                avg_cost += c / total_batch
            
            # Display logs
            if (epoch+1) % display_step == 0:
              print("Epoch:", '%04d' % (epoch+1), 
                    "cost=", "{:.9f}".format(avg_cost))
        
        print("Optimization Finished!")
        AUC = AUC_calc(sparse_tens_from_sparse_mat(test_vecs), test_target)
        print ("AUC_ROC:", AUC)
        auc_scores.append(AUC)
        
        sess.close()

save_auc_scores(auc_scores, APPROACH, CLASSIFIER, FLAVOR)
