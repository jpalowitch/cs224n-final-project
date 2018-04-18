from __future__ import print_function
from project_utils import *
#from diagnostics import *
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr
from sklearn.metrics import roc_auc_score

import os
import pandas as pd
import argparse
import random
import copy
random.seed(RUN_SEED)
parser = argparse.ArgumentParser(description='Run baseline comment models.')
parser.add_argument("-dataset", help="[toxic, attack]", type=str, default="toxic")
parser.add_argument("-weight_reg", help="regularization exponent = 3, 2, 1, 0: beta = 10**-weight_reg if nonzero, else is zero; default is 0 == no regularization",
type=int, default=0)
parser.add_argument("-sigmoid", help="do sigmoid model instead of per-class", action="store_true")
parser.add_argument("-nepochs", help="number of training epochs", type=int, default=50)
parser.add_argument("-batch_size", help="integer batch size", default=100, type=int)
args=parser.parse_args()

# Define params.
learning_rate = 0.001
display_step = 1
weight_reg = 10.0**(-args.weight_reg) * int(args.weight_reg > 0)

# Preprocess data.
if  args.dataset == 'attack':
    vecpath = TFIDF_VECTORS_FILE_AGG
    if not os.path.isfile(ATTACK_AGGRESSION_FN):
        get_and_save_talk_data()
    train, dev, test = get_TDT_split(
        pd.read_csv(ATTACK_AGGRESSION_FN, index_col=0).fillna(' '))
    cnames = train.columns.values[0:2]
    aucfn = "baseline_aucs.csv"
elif args.dataset == 'toxic':
    vecpath = TFIDF_VECTORS_FILE_TOXIC
    train, dev, test = get_TDT_split(
        pd.read_csv('train.csv').fillna(' '))
    cnames = CLASS_NAMES
    aucfn = "baseline_aucs.csv"
train_vecs, dev_vecs, test_vecs = vectorize_corpus_tf_idf(
    train, dev, test, sparse=True, path=vecpath, prot=2
)
n_train = train_vecs.shape[0]
if args.sigmoid:
    nclasses = len(cnames)
else:
    nclasses = 2

# Define graph input.
x = tf.sparse_placeholder(tf.float32)
y = tf.placeholder(tf.float32, [None, nclasses])

# Define layers.
W = tf.get_variable("weights",
                    shape=[NUM_FEATURES, nclasses],
                    initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.zeros([nclasses]))
theta = tf.sparse_tensor_dense_matmul(x, W) + b

# Get prediction (only used for testing).
pred = tf.nn.softmax(theta)

# Get cost directly (without needing prediction above).
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=theta, labels=y) + \
        tf.nn.l2_loss(W) * weight_reg
)

# Define optimizer.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
# Final scoring
def calc_auc_tf(X, Y, take_mean=True):
    if args.sigmoid:
        return calc_auc(Y, pred.eval({x: X}), take_mean)
    else:
        return calc_auc(Y[:, 1], pred.eval({x: X})[:, 1])    

# Making weight saving functionality
saver = tf.train.Saver()

# Initialize variables
global_init = tf.global_variables_initializer()

auc_scores = []

for target_class in range(len(cnames)):
    print("doing class " + cnames[target_class])
    
    # Get labels for training and save fn.
    save_fields = copy.deepcopy(vars(args))
    if args.sigmoid:
        train_target = train.values
        dev_target = dev.values
        test_target = test.values
        save_fn = saver_fn_rnn(save_fields, "sigmoid")
    else:
        train_labels = train[cnames[target_class]].values
        train_target = get_onehots_from_labels(train_labels)
        dev_labels = dev[cnames[target_class]].values
        dev_target = get_onehots_from_labels(dev_labels)
        test_labels = test[cnames[target_class]].values
        test_target = get_onehots_from_labels(test_labels)
        save_fn = saver_fn_rnn(save_fields, cnames[target_class])

    # Start training
    max_auc = 0
    with tf.Session() as sess:
    
        # Run initializer
        sess.run(global_init)
        
        # Training 
        for epoch in range(args.nepochs):
            avg_cost = 0.
            total_batch = int(n_train / args.batch_size)
            
            # Loop over batches
            minibatches = minibatch(train_vecs, train_target, args.batch_size)
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
        
        if args.sigmoid:
            auc_scores = calc_auc_tf(get_sparse_input(test_vecs), test_target, 
                                     mean=False)
        else:
            AUC = calc_auc_tf(get_sparse_input(test_vecs), test_target)
            print ("--Test AUC:", AUC)
            auc_scores.append(AUC)
        sess.close()
        
    if args.sigmoid:
        break
                
        AUC = calc_auc_tf(get_sparse_input(test_vecs), test_target)
        print ("Test AUC:", AUC)
        auc_scores.append(AUC)

        curr_pred = pred.eval({x: get_sparse_input(test_vecs)})
        pred_mat = np.column_stack((pred_mat, curr_pred[:,1]))
        target_mat = np.column_stack((target_mat, test_target[:,1]))
        
        sess.close()

fields = vars(args)
dataset = fields.pop('dataset')
save_auc_scores(auc_scores, fields, dataset, cnames)

