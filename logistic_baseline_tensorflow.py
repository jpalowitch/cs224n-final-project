from __future__ import print_function
from project_utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr
from sklearn.metrics import roc_auc_score
from sys import argv

import os
import pandas as pd

# Getting command args
# -nettype: either 'zero' or 'one', giving the number of hidden layers
myargs = getopts(argv)

APPROACH = "ngram"
CLASSIFIER = "logistic"
FLAVOR = "tensorflow-ADAM"

# Parameters
learning_rate = 0.001
hidden_size = 256
batch_size = 100
display_step = 1
dropout_rate = 0.5

# Get data and featurizing
train, dev, test = get_TDT_split(pd.read_csv('train.csv').fillna(' '))
train_vecs, dev_vecs, test_vecs = vectorize_corpus_tf_idf(
    train, dev, test, sparse=True
)
n_train = train_vecs.shape[0]
if batch_size is None:
    batch_size = train.shape[0]

# tf Graph Input
#x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
x = tf.sparse_placeholder(tf.float32)
y = tf.placeholder(tf.float32, [None, 2])

# Constructing middle layers
if myargs['-nettype'] == 'zero':
    beta_reg = 0.0001
    training_epochs = 50
    W = tf.get_variable("weights",
                        shape=[NUM_FEATURES, 2],
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.zeros([2]))
    theta = tf.sparse_tensor_dense_matmul(x, W) + b
elif myargs['-nettype'] == 'one':
    beta_reg = 0.0001
    FLAVOR = "tensorflow-ADAM-1layer"
    training_epochs = 10
    W = tf.get_variable("weights",
                        shape=[NUM_FEATURES, hidden_size],
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.zeros([hidden_size]))
    z = tf.sparse_tensor_dense_matmul(x, W) + b
    h = tf.nn.relu(z)
    h_drop = tf.nn.dropout(h, 1 - dropout_rate)
    W2 = tf.get_variable("weights2",
                     shape=[hidden_size, 2],
                     initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.zeros([2]))
    theta = tf.matmul(h_drop, W2) + b2

# Get prediction (this will only be used for testing)
pred = tf.nn.softmax(theta)

# Get cost directly (without needing prediction above)
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=theta, labels=y) + \
        tf.nn.l2_loss(W) * beta_reg
)

# Gradient Descent
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Final scoring
def calc_auc_tf(X, Y): 
    return calc_auc(Y[:, 1], pred.eval({x: X})[:, 1])

# Making weight saving functionality
saver = tf.train.Saver()

# Initialize the variables (i.e. assign their default value)
global_init = tf.global_variables_initializer()


auc_scores = []
pred_mat = np.ndarray(shape = (31915,0))
target_mat = np.ndarray(shape = (31915,0))
for target_class in range(6):
    print("doing class " + CLASS_NAMES[target_class])
    
    # Getting labels for training
    train_labels = train[CLASS_NAMES[target_class]].values
    train_target = get_onehots_from_labels(train_labels)
    dev_labels = dev[CLASS_NAMES[target_class]].values
    dev_target = get_onehots_from_labels(dev_labels)
    test_labels = test[CLASS_NAMES[target_class]].values
    test_target = get_onehots_from_labels(test_labels)
    
    # Getting weight saver fn
    save_fn = saver_fn(APPROACH, CLASSIFIER, FLAVOR, CLASS_NAMES[target_class])

    # Start training
    max_auc = 0
    with tf.Session() as sess:
    
        # Run initializer
        sess.run(global_init)
        
        # Training 
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_train/batch_size)
            
            # Loop over batches
            minibatches = minibatch(train_vecs, train_target, batch_size)
            for batch_xs_mat, batch_ys in minibatches:
                batch_xs = get_sparse_input(batch_xs_mat)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                avg_cost += c / total_batch
            
            # Display logs
            if (epoch+1) % display_step == 0:
                AUC = calc_auc_tf(get_sparse_input(dev_vecs), dev_target)
                print("Epoch:", '%04d' % (epoch+1), 
                      "cost=", avg_cost,
                      "dev.auc=", AUC)
                if AUC > max_auc:
                    print ("New best AUC on dev!")
                    saver.save(sess, save_fn)
                    max_auc = AUC
        
        print("Optimization Finished!")
        saver.restore(sess, save_fn)
        AUC = calc_auc_tf(get_sparse_input(test_vecs), test_target)
        print ("Test AUC:", AUC)
        auc_scores.append(AUC)

        curr_pred = pred.eval({x: get_sparse_input(test_vecs)})
        pred_mat = np.column_stack((pred_mat, curr_pred[:,1]))
        target_mat = np.column_stack((target_mat, test_target[:,1]))
        
        sess.close()
pred_mat = np.column_stack((pred_mat, target_mat))
diagnostics = Diagnostics(build = 'tf', model_type = 'logistic-zero', preds_targets = pred_mat)
diagnostics.do_all_diagnostics()
save_auc_scores(auc_scores, APPROACH, CLASSIFIER, FLAVOR)

