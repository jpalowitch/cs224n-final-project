import pandas as pd 
import numpy as np 

from keras.preprocessing.text import Tokenizer 
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback 
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout
from keras import backend as K 
from keras.models import Model 

from sklearn.metrics import roc_auc_score
from project_utils import * 

MAX_WORD_LENGTH = 10
MAX_WORDS = 15
MAX_NUM_CHARS = 1000
EMBEDDING_DIM = 10
DEV_SPLIT = 0.2

TRAIN = "train.csv"



def dot_product(x, kernel):
	return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)

class Attention(Layer):

	def __init__(self, W_reg=None, u_reg=None, b_reg=None,
					W_constraint=None, u_constraint=None, b_constraint=None,
					bias=True, **kwargs):

		self.supports_masking = True
		self.init = initializers.get('glorot_uniform')

		self.W_reg = regularizers.get(W_reg)
		self.u_reg = regularizers.get(u_reg)
		self.b_reg = regularizers.get(b_reg)

		self.W_constraint = constraints.get(W_constraint)
		self.u_constraint = constraints.get(u_constraint)
		self.b_constraint = constraints.get(b_constraint)

		self.bias = bias
		super(Attention, self).__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape) == 3

		self.W = self.add_weight((input_shape[-1], input_shape[-1],),
									initializer=self.init,
									name='{}_W'.format(self.name),
									regularizer=self.W_reg,
									constraint=self.W_constraint)

		if self.bias:
			self.b = self.add_weight((input_shape[-1],),
										initializer='zero',
										name='{}_b'.format(self.name),
										regularizer=self.b_reg,
										constraint=self.b_constraint)

		self.u = self.add_weight((input_shape[-1],),
									initializer=self.init,
									name='{}_u'.format(self.name),
									regularizer=self.u_reg,
									constraint=self.u_constraint)

		super(Attention, self).build(input_shape)

	def compute_mask(self, input, input_mask=None):

		return None

	def call(self, x, mask=None):
		uit = dot_product(x, self.W)

		if self.bias:
			uit += self.b 

		uit = K.tanh(uit)

		ait = dot_product(uit, self.u)

		a = K.exp(ait)

		if mask is not None:

			a *= K.cast(mask, K.floatx())

		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
		a = K.expand_dims(a)
		weighted_input = x * a 
		return K.sum(weighted_input, axis=1)

	def compute_output_shape(self, input_shape):
		return input_shape[0], input_shape[-1]


def preprocess_chars(data_train):
	classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
	comments = data_train["comment_text"]
	sentences = comments.apply(lambda x: x.split())
	tokenizer = Tokenizer(num_words=MAX_NUM_CHARS, char_level=True)
	tokenizer.fit_on_texts(sentences.values)
	data = np.zeros((len(sentences), MAX_WORDS, MAX_WORD_LENGTH), dtype='int32')
	for i, words in enumerate(sentences):
	for j, word in enumerate(words):
		if j < MAX_WORDS:
			k = 0
			for _, char in enumerate(word):
				try:
					if k < MAX_WORD_LENGTH:
						if tokenizer.word_index[char] < MAX_NUM_CHARS:
							data[i,j,k] = tokenizer.word_index[char]
							k=k+1
				except:
					None
	char_index = tokenizer.word_index
	return data, char_index
	

def split(data_train):
	labels = data_train[classes].values 
	indices = np.arange(data.shape[0])
	np.random.shuffle(indices)
	data = data[indices]
	labels = labels[indices]
	nb_dev_samps = int(DEV_SPLIT * data.shape[0])
	nb_train_samps = data.shape[0] - nb_dev_samps
	x_train = data[:-nb_dev_samps]
	y_train = labels[:-nb_dev_samps]
	x_dev = data[-nb_dev_samps:] 
	y_dev = labels[-nb_dev_samps:]
	return x_train, y_train, x_dev, y_dev




class RocAucEvaluation(Callback):
	def __init__(self, dev_data=(), interval=1):
		super(Callback, self).__init__()

		self.interval = interval
		self.X_dev, self.y_dev = dev_data

	def on_epoch_end(self, epoch, logs={}):
		if epoch % self.interval == 0:
			y_pred = self.model.predict(self.X_dev, verbose=0)
			score = roc_auc_score(self.y_dev, y_pred)
			print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))


def build_model(chars):
	embedding_layer = Embedding(len(char_index) + 1,
							EMBEDDING_DIM,
							input_length=MAX_WORD_LENGTH,
							trainable=True)
	#build character rnn
	char_input = Input(shape=(MAX_WORD_LENGTH,), dtype='int32')
	char_sequences = embedding_layer(char_input)
	char_lstm = Bidirectional(GRU(100, return_sequences=True))(char_sequences)
	char_dense = TimeDistributed(Dense(200))(char_lstm)
	char_dense = Dropout(0.5)(char_dense)
	#attention for character level
	char_att = Attention()(char_dense)
	charEncoder = Model(char_input, char_att)

	words_input = Input(shape=(MAX_WORDS, MAX_WORD_LENGTH), dtype='int32')
	words_encoder = TimeDistributed(charEncoder)(words_input)
	words_lstm = Bidirectional(GRU(100, return_sequences=True))(words_encoder)
	words_dense = TimeDistributed(Dense(200))(words_lstm)
	words_dense = Dropout(0.5)(words_dense)
	words_att = Attention()(words_dense)
	preds = Dense(6, activation='sigmoid')(words_att)
	model = Model(words_input, preds)
	return model 


if __name__ == "__main__":
	char = preprocess_chars()


model.compile(loss='binary_crossentropy',
				optimizer='adam',
				metrics=['acc'])

RocAuc = RocAucEvaluation(dev_data=(x_dev, y_dev), interval=1)
model.fit(x_train, y_train, validation_data=(x_dev, y_dev),
			epochs=10, batch_size=100, callbacks=[RocAuc])




















