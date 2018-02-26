from __future__ import print_function
from project_utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr
from sklearn.metrics import roc_auc_score

import pandas as pd

APPROACH = "ngram"
CLASSIFIER = "sigmoid"
FLAVOR = "tensorflow-ADAM"

# Parameters
learning_rate = 0.01
training_epochs = 50
batch_size = 1000
display_step = 1

# Get data and featurizing
train, dev, test = get_TDT_split(pd.read_csv('train.csv').fillna(' '))
train_vecs, dev_vecs, test_vecs = vectorize_corpus_tf_idf(
    train, dev, test, sparse=True
)
n_train = train_vecs.shape[0]
if batch_size is None:
    batch_size = train.shape[0]

# tf Graph Input
x = tf.sparse_placeholder(tf.float32)
y = tf.placeholder(tf.float32, [None, 6])

# Set model weights
W = tf.get_variable("weights",
                    shape=[NUM_FEATURES, 6],
                    initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.zeros([6]))

# Final layer
theta = tf.sparse_tensor_dense_matmul(x, W) + b

# Get prediction (this will only be used for testing)
pred = tf.nn.sigmoid(theta)

# Get cost directly (without needing prediction above)
cost = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=theta, labels=y)
)

# Gradient Descent
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Final scoring
def calc_auc_tf(X, Y): 
    return calc_auc(Y[:, 0], pred.eval({x: X})[:, 0])

# Initialize the variables (i.e. assign their default value)
global_init = tf.global_variables_initializer()

auc_scores = []    
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
            batch_xs = get_sparse_input(train_vecs[i_indxs])
            batch_ys = train[CLASS_NAMES].iloc[i_indxs].as_matrix()
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            avg_cost += c / total_batch
        
        # Display logs
        if (epoch+1) % display_step == 0:
            pred_mat = pred.eval({x: get_sparse_input(dev_vecs)})
            AUC = calc_auc(dev[CLASS_NAMES].as_matrix(), pred_mat)
            print("Epoch:", '%04d' % (epoch+1), 
                  "cost=", avg_cost,
                  "dev.auc=", AUC)
    
    print("Optimization Finished!")

    # Computing final AUC scores
    pred_mat = pred.eval({x: get_sparse_input(test_vecs)})
    auc_scores = calc_auc(test[CLASS_NAMES].as_matrix(), pred_mat, mean=False)
    
    sess.close()

save_auc_scores(auc_scores, APPROACH, CLASSIFIER, FLAVOR)
