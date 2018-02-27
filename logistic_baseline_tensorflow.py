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
learning_rate = 0.001
training_epochs = 50
beta_reg = 0.0001
batch_size = 100
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
#x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
x = tf.sparse_placeholder(tf.float32)
y = tf.placeholder(tf.float32, [None, 2])

# Set model weights
W = tf.get_variable("weights",
                    shape=[NUM_FEATURES, 2],
                    initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.zeros([2]))

# Final layer
theta = tf.sparse_tensor_dense_matmul(x, W) + b

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

# Initialize the variables (i.e. assign their default value)
global_init = tf.global_variables_initializer()

auc_scores = []
for target_class in range(6):
    
    # Getting labels for training
    train_labels = train[CLASS_NAMES[target_class]].values
    train_target = get_onehots_from_labels(train_labels)
    dev_labels = dev[CLASS_NAMES[target_class]].values
    dev_target = get_onehots_from_labels(dev_labels)
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
        
        print("Optimization Finished!")
        AUC = calc_auc_tf(get_sparse_input(test_vecs), test_target)
        print ("AUC_ROC:", AUC)
        auc_scores.append(AUC)
        
        sess.close()

save_auc_scores(auc_scores, APPROACH, CLASSIFIER, FLAVOR)
