from __future__ import print_function
from project_utils import *
from rnn_cell import RNNCell
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr
from sklearn.metrics import roc_auc_score
from sys import argv
from tensorflow.python.ops import init_ops
from word_embeddings import *


from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import dot, add, multiply, Lambda # need for attention
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K

import os
import pandas as pd

# Getting command args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-embeds", help="[stock, ours]", type=str, default="stock")
parser.add_argument("-dataset", help="[toxic, attack]", type=str, default="toxic")
parser.add_argument("-cell", help="[gru, lstm]", type=str, default="gru")
parser.add_argument("-bd", help="add for bidirectional", action="store_true")
parser.add_argument("-attn", help="add for attention", action="store_true")
args=parser.parse_args()


APPROACH = "rnn"
CLASSIFIER = "logistic"
FLAVOR = "tensorflow-ADAM-kerasSeqs"

# Parameters
max_features = 10000 # Originally 30000
learning_rate = 0.001
hidden_size = 80
batch_size = 32
embed_size = 100
max_length = 50
display_step = 1
dropout_rate = 0.5
training_epochs = 3


# Preparing data
if args.dataset == 'attack':
    vecpath = TFIDF_VECTORS_FILE_AGG
    if not os.path.isfile(ATTACK_AGGRESSION_FN):
        get_and_save_talk_data()
    train, dev, test = get_TDT_split(
        pd.read_csv(ATTACK_AGGRESSION_FN, index_col=0).fillna(' '))
    cnames = train.columns.values[0:2]
    aucfn = "auc_scores_attack.csv"
elif args.dataset == 'toxic':
    vecpath = TFIDF_VECTORS_FILE_TOXIC
    train, dev, test = get_TDT_split(
        pd.read_csv('train.csv').fillna(' '))
    cnames = CLASS_NAMES
    aucfn = "auc_scores.csv"


X_train = train["comment_text"].fillna("fillna").values
y_train = train[cnames].values
X_dev = dev["comment_text"].fillna("fillna").values
y_dev = dev[cnames].values
X_test = test["comment_text"].fillna("fillna").values
y_test = test[cnames].values

# Getting embeddings
if args.embeds == 'stock':
    FLAVOR = FLAVOR + 'stockEmbeds'
    EMBEDDING_FILE = 'glove.6B/glove.6B.100d.txt' # Originally 300d
    # Getting embeddings
    X_train, X_dev, X_test, embedding_matrix = get_stock_embeddings(
        X_train, X_dev, X_test,
        embed_file=EMBEDDING_FILE, embed_size=embed_size,
        max_features=max_features)
elif args.embeds == 'ours':
    FLAVOR = FLAVOR + 'ourEmbeds'
    embedding_matrix, X_train = get_embedding_matrix_and_sequences()
    _, X_dev = get_embedding_matrix_and_sequences(data_set="dev")
    _, X_test = get_embedding_matrix_and_sequences(data_set="test")

# Padding sequences
x_train = sequence.pad_sequences(X_train, maxlen=max_length)
x_dev = sequence.pad_sequences(X_dev, maxlen=max_length)
x_test = sequence.pad_sequences(X_test, maxlen=max_length)
n_train = len(x_dev)


print("building graph")
# tf Graph Input
inputs = tf.placeholder(tf.int32, shape=(None, max_length))
labels = tf.placeholder(tf.int32, [None, 2])

# Defining embeddings into graph
embeddings = tf.Variable(embedding_matrix)
embeddings = tf.cast(embeddings, tf.float32)
embeddings = tf.nn.embedding_lookup(params=embeddings, ids=inputs)
x = tf.reshape(tensor=embeddings, shape=[-1, max_length, embed_size])

# Run RNN and sum final product over sequence dimension
cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
if args.bd:
    cell_bw = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
    xs, state = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, x, dtype=tf.float32)
    x = tf.concat(xs, axis=2)
else:
    x, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

xmax = tf.reduce_max(x, axis=1)
xmean = tf.reduce_mean(x, axis=1)
x = tf.concat([xmax, xmean], axis=1)

# Define final layer variables
U = tf.get_variable(name="U", shape=(2 * (1 + int(args.bd)) * hidden_size, 2),
                    initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable(name="b2", shape=(2),
                     initializer=tf.constant_initializer(0.0))

# Making prediction
logits = tf.matmul(x, U) + b2
pred = tf.nn.softmax(logits)

# Get loss
ces = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
cost = tf.reduce_mean(ces)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Final scoring
def calc_auc_tf(X, Y): 
    return calc_auc(Y[:, 1], pred.eval({inputs: X})[:, 1])

# Making weight saving functionality
saver = tf.train.Saver()

# Initialize the variables (i.e. assign their default value)
global_init = tf.global_variables_initializer()

print("training on 6 classes")

# Preparing training
X_tra = x_train
X_dev = x_dev
X_test = x_test
y_tra = y_train
y_dev = y_dev
y_test = y_test

auc_scores = []
for target_class in range(6):
    print("doing class " + CLASS_NAMES[target_class])
    
    # Getting labels for training
    train_target = get_onehots_from_labels(y_tra[:, target_class])
    dev_target = get_onehots_from_labels(y_dev[:, target_class])
    test_target = get_onehots_from_labels(y_test[:, target_class])
    
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
            minibatches = minibatch(X_tra, train_target, batch_size)
            for batch_xs, batch_ys in minibatches:
                _, c = sess.run([optimizer, cost], feed_dict={inputs: batch_xs,
                                                              labels: batch_ys})
                avg_cost += c / total_batch
            
            # Display logs
            if (epoch+1) % display_step == 0:
                AUC = calc_auc_tf(X_dev, dev_target)
                print("Epoch:", '%04d' % (epoch+1), 
                      "cost=", avg_cost,
                      "dev.auc=", AUC)
                if AUC > max_auc:
                    print ("New best AUC on dev!")
                    saver.save(sess, save_fn)
                    max_auc = AUC
        
        print("Optimization Finished!")
        saver.restore(sess, save_fn)
        AUC = calc_auc_tf(X_test, test_target)
        print ("Test AUC:", AUC)
        auc_scores.append(AUC)
        
        sess.close()

save_auc_scores(auc_scores, APPROACH, CLASSIFIER, FLAVOR)

