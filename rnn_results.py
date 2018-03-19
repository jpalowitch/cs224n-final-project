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
import sys
import copy
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
parser.add_argument("-batch_size", help="integer batch size", default=32, type=int)
parser.add_argument("-tag", help="score filename tag if you want to separately save these auc scores", type=str, default=None)
args=parser.parse_args()

if not args.sigmoid:
    print("only sigmoid-driven results are implemented")
    sys.exit

if args.gpu:
    device = "/gpu:0"
else:
    device = "/cpu:0"


APPROACH = "rnn"
CLASSIFIER = "logistic"
FLAVOR = "tensorflow-ADAM"

# Parameters
max_features = 30000 # Originally 30000
starter_learning_rate = 0.001 # starter learning rate for adaptive lr
learning_rate = 0.001 # used if -adapt_lr flag not present
lr_decay = 0.95
hidden_size = args.hidden_size
embed_size = 100
batch_size = args.batch_size
max_length = args.max_length
display_step = 1
dense_dropout = args.dense_drop / 100.0
embed_dropout = args.embed_drop / 100.0
weight_reg = 10.0**(-args.weight_reg) * int(args.weight_reg > 0)
training_epochs = args.nepochs

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
    X_train, X_dev, X_test, embedding_matrix, tknzr = get_stock_embeddings(
        X_train, X_dev, X_test,
        embed_file=EMBEDDING_FILE, embed_size=embed_size,
        max_features=max_features, return_tokenizer=True)
elif args.embeds == 'ours':
    FLAVOR = FLAVOR + 'ourEmbeds'
    embedding_matrix, X_train = get_embedding_matrix_and_sequences()
    _, X_dev = get_embedding_matrix_and_sequences(data_set="dev")
    _, X_test = get_embedding_matrix_and_sequences(data_set="test")

# Getting weight saver fn
save_fields = copy.deepcopy(vars(args))
save_fields.pop('embed_drop')
save_fields.pop('dense_drop')
save_fields.pop('gpu')
save_fn = saver_fn_rnn(save_fields, cnames[0])
save_fn_stem = saver_fn_rnn(save_fields, cnames[0], stem=True)
pkl_fn = 'data/' + save_fn_stem + "_results.pkl"
print("pickle_save_fn = " + pkl_fn)
with open(pkl_fn, "rb") as File:
    d = pickle.load(File)

alphas = d['alphas']
preds = d['preds']

print(alphas)
print(preds)

# Preparing comment sets
toxic_ind = np.count_nonzero(y_test, axis=1)
N = len(toxic_ind)
assert N == d['preds'].shape[0]
toxic_comments = test['comment_text'][toxic_ind > 0].values
safe_comments = test['comment_text'][toxic_ind == 0].values

print(toxic_comments[0])
print(safe_comments[0])

# Computing losses
def CE(yhat, y):
    return -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat), axis=1)
losses = CE(preds, y_test)
toxic_losses = losses[toxic_ind > 0]
safe_losses = losses[toxic_ind == 0]


top_toxic_wins = np.argsort(toxic_losses)[:500]
top_toxic_misses = np.argsort(toxic_losses)[-500:]
top_safe_misses = np.argsort(safe_losses)[-500:]
#print(toxic_comments[top_toxic_wins][:3])
#print(toxic_comments[top_toxic_misses][:3])
#print(safe_comments[top_safe_misses][:3])

# Getting tokenized sentence
wi = tknzr.word_index
wi2 = {y:x for x,y in wi.iteritems()}
def get_wordlist (comment):
    indx = tknzr.texts_to_sequences([comment])
    if len(indx) > max_length:
        indx = indx[:max_length]
    wordlist = [wi2[i] for i in indx[0] if i is not 0]
    return wordlist
def get_tknz_sent (comment):
    wordlist = get_wordlist(comment)
    return(" ".join(wordlist))


test_i = top_toxic_wins[0]
tknz_sent = get_tknz_sent(toxic_comments[test_i])

top_toxic_win_comments = pd.DataFrame(
    [get_tknz_sent(toxic_comments[i]) for i in top_toxic_wins],
    columns=['comment'])
top_toxic_miss_comments = pd.DataFrame(
    [get_tknz_sent(toxic_comments[i]) for i in top_toxic_misses],
    columns=['comment'])
top_safe_miss_comments = pd.DataFrame(
    [get_tknz_sent(safe_comments[i]) for i in top_safe_misses],
    columns=['comment'])

# getting prediction scores
toxic_preds = preds[toxic_ind > 0]
safe_preds = preds[toxic_ind == 0]
top_toxic_win_preds = toxic_preds[top_toxic_wins]
top_toxic_miss_preds = toxic_preds[top_toxic_misses]
top_safe_miss_preds = safe_preds[top_safe_misses]

twin_preds = pd.DataFrame(top_toxic_win_preds, columns=cnames).round(4)
tmiss_preds = pd.DataFrame(top_toxic_miss_preds, columns=cnames).round(4)
smiss_preds = pd.DataFrame(top_safe_miss_preds, columns=cnames).round(4)

twin_df = pd.concat([top_toxic_win_comments, twin_preds], axis=1)
tmiss_df = pd.concat([top_toxic_miss_comments, tmiss_preds], axis=1)
smiss_df = pd.concat([top_safe_miss_comments, smiss_preds], axis=1)

print(twin_df.head())
print(tmiss_df.head())
print(smiss_df.head())

# Getting top two attention words
talphas = alphas[toxic_ind > 0]
salphas = alphas[toxic_ind == 0]
def get_attn_words (comment, avec, nwords=5):
    wordlist = get_wordlist(comment)
    avec = avec[:len(wordlist)]
    windxs = np.argsort(avec)[-nwords:]
    attn_words = [wordlist[i] for i in list(windxs)]
    attn_words.reverse()
    return(" ".join(attn_words))
twin_attn_words = [get_attn_words(toxic_comments[i], talphas[i]) for i in top_toxic_wins]
tmiss_attn_words = [get_attn_words(toxic_comments[i], talphas[i]) for i in top_toxic_misses]
smiss_attn_words = [get_attn_words(safe_comments[i], salphas[i]) for i in top_safe_misses]

twin_df['attn_words'] = twin_attn_words
tmiss_df['attn_words'] = tmiss_attn_words
smiss_df['attn_words'] = smiss_attn_words

twin_df.to_csv(args.dataset + '_twin_results.csv')
tmiss_df.to_csv(args.dataset + '_tmiss_results.csv')
smiss_df.to_csv(args.dataset + '_smiss_results.csv')
    


