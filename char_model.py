from project_utils import *
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.callbacks import Callback
from keras.layers import Dense, Activation, Embedding, LSTM, Input, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import SpatialDropout1D, Bidirectional
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K
from keras.utils import plot_model
import pandas as pd 
import numpy as np 
import random
import sys 
import csv 
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-max_length", help="int: size of longest sequence", type=int, default=100)
parser.add_argument("-hidden_size", help="int: size of rnn layer", type=int, default=256)
parser.add_argument('-cell', help="[gru, lstm]", type=str, default='gru')
parser.add_argument('-optimizer', help="[rmsprop, adam]", type=str, default='adam')
args=parser.parse_args()


maxlen = args.max_length
batch_size = 128
embedding_dim = 300
epochs=3
lr_init = 0.001
lr_end = 0.0005



cnames = CLASS_NAMES
EMBEDDING_FILE = 'char_embeds.txt'

train, dev, test = get_TDT_split(pd.read_csv('train.csv').fillna(' '))
X_train = train["comment_text"].fillna("fillna")
y_train = train[cnames].values
X_dev = dev["comment_text"].fillna("fillna")
y_dev = dev[cnames].values
X_test = test["comment_text"].fillna("fillna")
y_test = test[cnames].values

chars, docs = preprocess_char(X_train) #set of chars from training data


char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

#prep data
X_tra = get_char_features(char_indices, X_train, maxlen) 
X_dev = get_char_features(char_indices, X_dev, maxlen)
X_test = get_char_features(char_indices, X_test, maxlen)

embedding_matrix = get_char_embeddings(char_indices)


print ('Building model...')

#logger for AUC
class RocAucEvaluation(Callback):
    def __init__(self, dev_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_dev, self.y_dev = dev_data

        def on_epoch_end(self, epoch, logs={}):
            if epoch % self.interval ==0:
                y_pred = self.model.predict(self.X_dev, verbose=0)
                score = roc_auc_score(self.y_dev, y_pred)
                print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

#logger for loss history
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs = {}):
        self.losses.append(logs.get('loss'))


def get_model():
    inputs = Input(shape=(maxlen,))
    embedding_layer = Embedding(len(chars), embedding_dim, input_length=maxlen,
                        weights=[embedding_matrix])
    embedded = embedding_layer(inputs)
    drop = SpatialDropout1D(0.2)(embedded)
    if args.cell == 'gru':
        x = Bidirectional(GRU(args.hidden_size, return_sequences=True))(drop)
    else:
        x = Bidirectional(LSTM(args.hidden_size, return_sequences=True))(drop)
    max_pool = GlobalMaxPooling1D()(x)
    out = Dense(6, activation="sigmoid")(max_pool)
    model = Model(inputs=inputs, outputs=out)
    if args.optimizer == 'rmsprop':
        model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
    else:
        model.compile(loss='binary_crossentropy',
                        optimizer='RMSprop',
                        metrics=['accuracy'])
    return model

model = get_model()

RocAuc = RocAucEvaluation(dev_data=(X_dev, y_dev), interval=1)
history = LossHistory()

exp_decay = lambda init, end, steps: (init/end)**(1/(steps-1)) - 1
steps = int(len(X_tra)/batch_size) * epochs
lr_decay = exp_decay(lr_init, lr_end, steps)
K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_end)
fit = model.fit(X_tra, y_train, validation_data=(X_dev,y_dev), 
            batch_size=batch_size, epochs=3, callbacks=[RocAuc,history], verbose=1)

y_preds = model.predict(X_test, batch_size=1024, verbose=1)

print ("Mean AUC: ", calc_auc(y_test, y_preds))

AUC = calc_auc(y_test,y_preds, mean=False)

filename = 'data/auc_scores_' + 'chars' + str(args.cell) + str(args.max_length) + '.csv'

save_auc_scores(AUC, "lstm_char250max", "sigmoid", "Adam",
                    fn=filename, overwrite=True, cnames=CLASS_NAMES)

with open('data/char_lstm250max_sigmoid_model_chars3', 'wb') as file_pi:
        pickle.dump(fit.history, file_pi)
