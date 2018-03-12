import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import dot, add, multiply, Lambda # need for attention
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K

from project_utils import *
from sys import argv

# Getting command args
# -nettype: either 'zero' or 'one', giving the number of hidden layers
# -dataset: either 'toxic' or 'attack', telling which data set to analyze
myargs = getopts(argv)

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

APPROACH = 'GRU-perclass'
CLASSIFIER = 'logistic'
FLAVOR = 'pooled'

EMBEDDING_FILE = 'glove.6B/glove.6B.50d.txt' # Originally 300d

#submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

if myargs['-dataset'] == 'attack':
    vecpath = TFIDF_VECTORS_FILE_AGG
    if not os.path.isfile(ATTACK_AGGRESSION_FN):
        get_and_save_talk_data()
    train, dev, test = get_TDT_split(
        pd.read_csv(ATTACK_AGGRESSION_FN, index_col=0).fillna(' '))
    cnames = train.columns.values[0:2]
    aucfn = "auc_scores_attack.csv"
elif myargs['-dataset'] == 'toxic':
    vecpath = TFIDF_VECTORS_FILE_TOXIC
    train, dev, test = get_TDT_split(
        pd.read_csv('train.csv').fillna(' '))
    cnames = CLASS_NAMES
    aucfn = "auc_scores.csv"


X_train = train["comment_text"].fillna("fillna").values
y_train = train[cnames].values
X_val = dev["comment_text"].fillna("fillna").values
y_val = dev[cnames].values
X_test = test["comment_text"].fillna("fillna").values


max_features = 10000 # Originally 30000
maxlen = 50 # Originally 100
embed_size = 50 # Originally 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_val = sequence.pad_sequences(X_val, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


def get_coefs(word, *arr): 
    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        
        self.interval = interval
        self.X_val, self.y_val = validation_data
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
    
    def get_score(self):
        y_pred = self.model.predict(self.X_val, verbose=0)
        score = roc_auc_score(self.y_val, y_pred)
        return score

def rowise_matmul(a, b, add_axis=1, sum_axis=2):
    """ Expands b in a middle axis, takes element-wise product with a,
        and reduce_sums the product along the middle axis.
    Args:
        a: [None, M, H] tensor
        b: [None, M] tensor or [None, H] tensor
    Returns:
        the axis-1 sum of the element-wise product of a and b_star = b expanded
        to dimension of a
    """
    b_star = K.repeat(b, K.int_shape(a)[add_axis])
    if add_axis == 2:
        b_star = K.permute_dimensions(b_star, (0, 2, 1))
    return K.sum(multiply([a, b_star]), axis=sum_axis)

def get_attention_output(tens_list):
    """ Explanatory, see below
    """
    attention_avg = rowise_matmul(tens_list[0], tens_list[1])
    attention_max = rowise_matmul(tens_list[0], tens_list[2])
    attention_softmax = K.softmax(add([attention_avg, attention_max]))
    return rowise_matmul(tens_list[0], attention_softmax, 2, 1)

def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    if myargs['-attention'] == 'yes':
        attention_output = Lambda(get_attention_output)([x, avg_pool, max_pool])            
        conc = concatenate([avg_pool, max_pool, attention_output])
    elif myargs['-attention'] == 'no':
        conc = concatenate([avg_pool, max_pool])
    outp = Dense(2, activation="softmax")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model


# Preparing training
X_tra = x_train
X_val = x_val
y_tra = y_train
y_val = y_val
#X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
batch_size = 32
nepochs = 3


auc_scores = []
for target_class in range(len(cnames)):
    
    print("doing class " + cnames[target_class])
    train_target = get_onehots_from_labels(y_tra[:, target_class])
    dev_target = get_onehots_from_labels(y_val[:, target_class])
    RocAuc = RocAucEvaluation(validation_data=(X_val, dev_target), interval=1)
    model = get_model()
    
    # Fitting and predicting
    hist = model.fit(X_tra, train_target, batch_size=batch_size, epochs=nepochs, 
                     validation_data=(X_val, dev_target),
                     callbacks=[RocAuc], verbose=2)
    aucscore = RocAuc.get_score()
    print("--auc is: ", aucscore)
    auc_scores.append(RocAuc.get_score())

if myargs['-attention'] == 'yes':
    FLAVOR = FLAVOR + '-attention'

save_auc_scores(auc_scores, APPROACH, CLASSIFIER, FLAVOR, 
                fn=aucfn, cnames=cnames)
