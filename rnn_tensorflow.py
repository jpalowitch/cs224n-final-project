from __future__ import print_function
from project_utils import *
from rnn_cell import RNNCell
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr
from sklearn.metrics import roc_auc_score
from sys import argv
from tensorflow.python.ops import init_ops
from word_embeddings import *

from keras.preprocessing import text, sequence
from keras.callbacks import Callback

import os
import pandas as pd
import random
random.seed(RUN_SEED)

# Getting command args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-embeds", help="[stock, ours]", type=str, default="stock")
parser.add_argument("-dataset", help="[toxic, attack]", type=str, default="toxic")
parser.add_argument("-cell", help="[gru, lstm]", type=str, default="gru")
parser.add_argument("-bd", help="add for bidirectional", action="store_true")
parser.add_argument("-attn", help="add for attention", action="store_true")
parser.add_argument("-embed_drop", help="dropout probability for embeddings = integer in [0, 100]; default 0", default=0, type=int)
parser.add_argument("-dense_drop", help="dropout probability for final dense layer = integer in [0, 100]; default 0", default=0, type=int)
parser.add_argument("-weight_reg", help="regularization exponent = 3, 2, 1, 0: beta = 10**-weight_reg if nonzero, else is zero; default is 0 == no regularization",
type=int, default=0)
parser.add_argument("-nepochs", help="number of training epochs", type=int, default=3)
parser.add_argument("-sigmoid", help="do sigmoid model instead of per-class", action="store_true")
parser.add_argument("-gpu", help="add to use gpu on azure", action="store_true")
parser.add_argument("-hidden_size", help="int: size of rnn layer", default=80, type=int)
parser.add_argument("-max_length", help="int: size of longest sequence", default=50, type=int)
parser.add_argument("-adapt_lr", help="add if you want learning rate to be adaptive", action="store_true")
args=parser.parse_args()

if args.gpu:
    device = "/gpu:0"
else:
    device = "/cpu:0"


APPROACH = "rnn"
CLASSIFIER = "logistic"
FLAVOR = "tensorflow-ADAM-kerasSeqs"

# Parameters
max_features = 10000 # Originally 30000
starter_learning_rate = 0.001 # starter learning rate for adaptive lr
learning_rate = 0.001 # used if -adapt_lr flag not present
hidden_size = args.hidden_size
batch_size = 32
embed_size = 100
max_length = args.max_length
display_step = 1
dense_dropout = args.dense_drop / 100.0
embed_dropout = args.embed_drop / 100.0
weight_reg = 10.0**(-args.weight_reg) * int(args.weight_reg > 0)
training_epochs = args.nepochs


with tf.device(device):

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

    if args.sigmoid:
        nclasses = len(cnames)
    else:
        nclasses = 2

    X_train = train["comment_text"].fillna("fillna").values
    y_train = train[cnames].values
    X_dev = dev["comment_text"].fillna("fillna").values
    y_dev = dev[cnames].values
    X_test = test["comment_text"].fillna("fillna").values
    y_test = test[cnames].values

    # Getting embeddings
    if args.embeds == 'stock':
        FLAVOR = FLAVOR + 'stockEmbeds'
        EMBEDDING_FILE = 'data/glove.6B.100d.txt' # Originally 300d
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
    padder = lambda z: sequence.pad_sequences(
        z, padding='post', truncating='post', maxlen=max_length)
    x_train = padder(X_train)
    x_dev = padder(X_dev)
    x_test = padder(X_test)
    n_train = len(x_dev)


    print("building graph")
    # tf Graph Input
    inputs = tf.placeholder(tf.int32, shape=(None, max_length))
    labels = tf.placeholder(tf.int32, [None, nclasses])
    seq_lengths = tf.placeholder(tf.int32, [None])

    # Defining embeddings into graph
    embeddings = tf.Variable(embedding_matrix)
    embeddings = tf.cast(embeddings, tf.float32)
    embeddings = tf.nn.embedding_lookup(params=embeddings, ids=inputs)
    x = tf.reshape(tensor=embeddings, shape=[-1, max_length, embed_size])
    x = tf.nn.dropout(x, keep_prob=1.0 - embed_dropout,
                   noise_shape=[1, 1, embed_size])
    # Run RNN and sum final product over sequence dimension
    if args.cell == 'gru':
        cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
        cell_bw = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
    elif args.cell == 'lstm':
        cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)

    if args.bd:
        xs, state = tf.nn.bidirectional_dynamic_rnn(
            cell, cell_bw, x, sequence_length=seq_lengths, dtype=tf.float32)
        outputs = tf.concat(xs, axis=2)
    else:
        outputs, state = tf.nn.dynamic_rnn(
            cell, x, sequence_length=seq_lengths, dtype=tf.float32)

    xmax = tf.reduce_max(outputs, axis=1)
    xmean = tf.reduce_mean(outputs, axis=1)
    xpool = tf.concat([xmax, xmean], axis=1)
    true_hidden_size = (1 + int(args.bd)) * hidden_size
    pooled_size = 2 * true_hidden_size

    if args.attn:
        W = tf.get_variable(name="attnW", shape=(true_hidden_size, true_hidden_size),
                            initializer=tf.contrib.layers.xavier_initializer())
        def get_attention_output(z):
            right_side = tf.matmul(z, W) # batch_size x true_hidden_size
            right_side_broadcast = tf.expand_dims(right_side, 1) # batch_size x 1 x true_hidden_size
            threeDeeMult = outputs * right_side_broadcast # batch_size x max_length x true_hidden_size
            alpha_logits = tf.reduce_sum(threeDeeMult, 2) # batch_size x max_length
            alphas = tf.nn.softmax(alpha_logits) # batch_size x max_length
            alphas_broadcast = tf.expand_dims(alphas, 2) # batch_size x max_length x 1
            outputs_weighted = outputs * alphas_broadcast # batch_size x max_length x true_hidden_size
            return tf.reduce_sum(outputs_weighted, axis=1) # batch_size x true_hidden_size
        amax = get_attention_output(xmax)
        amean = get_attention_output(xmean)
        xpool = tf.concat([xpool, amax, amean], axis=1)
        pooled_size = 2 * pooled_size

    # Define final layer variables
    xpool = tf.nn.dropout(xpool, keep_prob=1.0 - dense_dropout, noise_shape=[1, pooled_size])
    U = tf.get_variable(name="U", shape=(pooled_size, nclasses),
                        initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable(name="b2", shape=(nclasses),
                         initializer=tf.constant_initializer(0.0))

    # Making prediction
    logits = tf.matmul(xpool, U) + b2
    if args.sigmoid:
        pred = tf.nn.sigmoid(logits)
    else:
        pred = tf.nn.softmax(logits)

    # Get loss
    if args.sigmoid:
        labels = tf.cast(labels, dtype=tf.float32)
        ces = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    else:
        ces = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
    cost = tf.reduce_mean(ces) + weight_reg * tf.nn.l2_loss(U)

    # Setting learning rate
    if args.adapt_lr:
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   500, 0.90, staircase=True)
    else:
        learning_rate = learning_rate #tf.constant(learning_rate)
    # Optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate)
    if args.adapt_lr:
        optimizer = train_op.minimize(cost, global_step)
    else:
        optimizer = train_op.minimize(cost)

    # Final scoring
    def calc_auc_tf(X, Y, seq_lens, mean=True):
        if args.sigmoid: 
            return calc_auc(Y, pred.eval({inputs: X, seq_lengths: seq_lens}), mean)
        else:
            return calc_auc(Y[:, 1], pred.eval({inputs: X, seq_lengths: seq_lens})[:, 1])

    # Initialize the variables (i.e. assign their default value)
    global_init = tf.global_variables_initializer()

    if args.sigmoid:
        print("training on all classes simultaneously")
    else:
        print("training on 6 classes")

    # Making weight saving functionality
    #saver = tf.train.Saver()

    # Preparing training
    X_tra = x_train
    X_dev = x_dev
    X_test = x_test
    y_tra = y_train
    y_dev = y_dev
    y_test = y_test

    auc_scores = []
    current_lr = starter_learning_rate
    dev_lengths = np.count_nonzero(X_dev, axis=1)
    test_lengths = np.count_nonzero(X_test, axis=1)
    for target_class in range(6):
        print("doing class " + cnames[target_class])
        
        # Getting labels for training
        if args.sigmoid:
            train_target = y_tra
            dev_target = y_dev
            test_target = y_test
        else:
            train_target = get_onehots_from_labels(y_tra[:, target_class])
            dev_target = get_onehots_from_labels(y_dev[:, target_class])
            test_target = get_onehots_from_labels(y_test[:, target_class])
                
        # Getting weight saver fn
        save_fn = saver_fn_rnn(vars(args), cnames[target_class])

        # Start training
        max_auc = 0
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        
            # Run initializer
            saver = tf.train.Saver()
            sess.run(global_init)
            
            # Training 
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(n_train/batch_size)
                
                # Loop over batches
                minibatches = minibatch(X_tra, train_target, batch_size)
                for batch_xs, batch_ys in minibatches:
                    batch_lengths = np.count_nonzero(batch_xs, axis=1)
                    _, c = sess.run([optimizer, cost], feed_dict={
                        inputs: batch_xs, 
                        labels: batch_ys, 
                        seq_lengths: batch_lengths})
                    avg_cost += c / total_batch
                
                # Display logs
                if (epoch+1) % display_step == 0:
                    AUC = calc_auc_tf(X_dev, dev_target, dev_lengths)
                    print("Epoch:", '%04d' % (epoch+1), 
                          "cost=", avg_cost,
                          "dev.auc=", AUC)
                    if AUC > max_auc:
                        print ("New best AUC on dev!")
                        saver.save(sess, save_fn)
                        max_auc = AUC
            
            print("Optimization Finished!")
            saver.restore(sess, save_fn)
            if args.sigmoid:
                auc_scores = calc_auc_tf(X_test, test_target, test_lengths, mean=False)
                print (auc_scores)
                print (type(auc_scores))
            else:
                AUC = calc_auc_tf(X_test, test_target, test_lengths)
                print ("Test AUC:", AUC)
                auc_scores.append(AUC)        
            sess.close()
        if args.sigmoid:
            break

    fields = vars(args)
    fields.pop('gpu')
    dataset = fields.pop('dataset')
    save_rnn_auc_scores(auc_scores, fields, dataset, cnames)

