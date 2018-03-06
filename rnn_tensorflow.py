from __future__ import print_function
from project_utils import *
from word_embeddings import *
from rnn_cell import RNNCell
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr
from sklearn.metrics import roc_auc_score
from sys import argv

import os
import pandas as pd

# Getting command args
# -nettype: either 'zero' or 'one', giving the number of hidden layers
myargs = getopts(argv)

APPROACH = "rnn"
CLASSIFIER = "logistic"
FLAVOR = "tensorflow-ADAM"

# Parameters
learning_rate = 0.001
beta_reg = 0.0001
hidden_size = 256
batch_size = 100
embed_size = 50
max_length = 100
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
train_inputs = get_embedding



# tf Graph Input
inputs = tf.placeholder(tf.int32, shape=(None, max_length))
mask = tf.placeholder(tf.bool, shape=(None, max_length))
labels = tf.placeholder(tf.int32, [None, 2])

# Getting embeddings
embeddings = tf.Variable(pretrained_embeddings)
embeddings = tf.nn.embedding_lookup(params=embeddings, ids=inputs)
x = tf.reshape(tensor=embeddings, shape=[-1, max_length, embed_size])

# RNN variables and cell
U = tf.get_variable(name="U", shape=(hidden_size, 2),
                    initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable(name="b2", shape=(2),
                     initializer=tf.constant_initializer(0))
h_t = tf.zeros((tf.shape(x)[0], hidden_size))
cell = RNNCell(Config.embed_size, Config.hidden_size)

# RNN training
hlayers = []
with tf.variable_scope("RNN"):
    for time_step in range(max_length):
        if time_step > 0:
            tf.get_variable_scope().reuse_variables()
        o_t, h_t = cell(x[:, time_step, :], h_t, scope="RNN")
        o_drop_t = tf.nn.dropout(o_t, dropout_rate)
        hlayers.append(o_drop_t)

# Sum the hidden layers and predict on that    
hlayers_sum = tf.add_n(hlayers) 
logits = tf.add(tf.matmul(hlayers_sum, U), b2)
pred = tf.nn.softmax(logits)

# Get loss
ces = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.boolean_mask(logits, mask), 
            labels=tf.boolean_mask(labels, mask)
)
loss = tf.reduce_mean(ces)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Final scoring
def calc_auc_tf(X, Y): 
    return calc_auc(Y[:, 1], pred.eval({x: X})[:, 1])

# Making weight saving functionality
saver = tf.train.Saver()

# Initialize the variables (i.e. assign their default value)
global_init = tf.global_variables_initializer()


auc_scores = []
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
                _, c = sess.run([optimizer, cost], feed_dict={inputs: batch_xs,
                                                              mask: batch_mask,
                                                              labels: batch_ys})
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
        
        sess.close()

save_auc_scores(auc_scores, APPROACH, CLASSIFIER, FLAVOR)

