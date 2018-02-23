import numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from string import punctuation 
from nltk import word_tokenize
import pickle 

path = "~/Google Drive/" #directory on emily's laptop
KAGGLE_TRAIN = pd.read_csv(path + "train.csv")
COMMENT = 'comment_text'
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
PUNCTUATION = [punctuation[i:i+1] for i in range(0, len(punctuation), 1)]

def split_data(KAGGLE_TRAIN):
	'''
	splits kaggle's train data into ratio of 60-20-20 train / dev / test
	adds a 'none' label
	there are a few NAs, replaces these with "unknown" which is usually a little
	happier for sklearn models
	'''
	X = KAGGLE_TRAIN.iloc[:,:2]
	y = KAGGLE_TRAIN.iloc[:,2:]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
	random_state=94110)
	X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5,
	random_state=94110)
	train = pd.concat([X_train, y_train],axis=1)
	dev = pd.concat([X_dev,y_dev],axis=1)
	test = pd.concat([X_test,y_test],axis=1)
	for d in [train, dev, test]:
		#d['none'] = 1 - d[LABELS].max(axis=1)
		d[COMMENT].fillna("unknown", inplace=True)
	return train, dev, test

def tokenize(comment):
	'''
	for one comment, tokenizes, removes punctuation and changes to lowercase
	'''
	words = word_tokenize(comment)
	words = [w.lower() for w in words]
	words = [w for w in words if w not in PUNCTUATION and not w.isdigit()]
	return words


def createDocTermMatrices(train, dev, test):
	'''
	creates TFidf document term matrix from pandas dataframe where 2nd column 
	contains comments, 
	removes punctuation from sentences and converts to lowercase
	'''
	vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1)
	train_dtm = vec.fit_transform(train[COMMENT])
	dev_dtm = vec.transform(dev[COMMENT])
	test_dtm = vec.transform(test[COMMENT])
	return train_dtm, dev_dtm, test_dtm

def model_one_label(dtm, label):
	clf = MultinomialNB() #also test other versions and see which works best
	return clf.fit(dtm, label)



def naive_bayes(dtm_train, dtm_test, y_train):
	'''
	return prediction probabilitities for each class
	'''
	pred_mat = np.zeros((dtm_test.shape[0], y_train.shape[1]))
	for i, j in enumerate(LABELS):
		print('fit',j)
		mod = model_one_label(dtm_train, y_train[j])
		pred_mat[:,i] = mod.predict_proba(dtm_test)[:,1]
	return pred_mat



if __name__ == "__main__":
	train, dev, test = split_data(KAGGLE_TRAIN)
	print "creating doc term matrices..."
	train_dtm, dev_dtm, test_dtm = createDocTermMatrices(train, dev, test)
	print "successfully created doc term matrices"
	y_train = train.iloc[:,2:]
	y_dev = dev.iloc[:,2:]
	print "running naive bayes model..."
	preds = naive_bayes(train_dtm, dev_dtm, y_train)
	auc = roc_auc_score(y_dev, preds) 
	print "auc-roc: " +str(auc) #dev AUC SCORE: 0.836970136328
	








