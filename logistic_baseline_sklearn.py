import numpy as np
import pandas as pd

from project_utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.sparse import hstack
from sys import argv
from word_embeddings import get_tokenized_sentences

APPROACH = "ngram"
CLASSIFIER = "logistic"
FLAVOR = "sklearn-SAG"

use_local_embeddings = False
path = ""
dims = 50

def fit_logistic_ngram_sklearn(train, dev, test, dataset="toxic",
                               cnames=CLASS_NAMES):
    """Fits and evals ngram logistic regression model with sklearn package.

    Args:
      train: a pd.DataFrame of the training data.
      dev: a pd.DataFrame of the dev data.
      vocab:
    Returns:
      average ROC-AUC score over classes.
    """

    if dataset == "toxic":
        vecpath = TFIDF_VECTORS_FILE_TOXIC
    elif dataset == "attack":
        vecpath = TFIDF_VECTORS_FILE_AGG

    if use_local_embeddings:
        sentences = get_tokenized_sentences(use_local=True, load_files=False, \
            local_embeddings_path=path, n_dims=dims)
        train_vecs = sentences.get("train").get("vectors")
        test_vecs = sentences.get("test").get("vectors")
        dev_vecs = sentences.get("dev").get("vectors")
    else:
        train_vecs, dev_vecs, test_vecs = \
            vectorize_corpus_tf_idf(train, dev, test, sparse=True, path=vecpath)

    # Doing one-vs-all training
    auc_scores = []
    for class_name in [cnames[x] for x in range(len(cnames))]:
        print('doing class {}'.format(class_name))

        # Training model
        train_target = train[class_name]
        classifier = LogisticRegression(solver='sag')
        model = classifier.fit(train_vecs, train_target)

        # Computing ROC
        test_pred = model.predict_proba(test_vecs)
        test_target = get_onehots_from_labels(test[class_name].values)
        ROC_AUC_score = roc_auc_score(test_target, test_pred)
        auc_scores.append(ROC_AUC_score)
        print('--AUC score is {}'.format(ROC_AUC_score))

    return auc_scores


if __name__ == '__main__':
    myargs = getopts(argv)
    if  myargs['-dataset'] == 'attack':
        if not os.path.isfile(ATTACK_AGGRESSION_FN):
            get_and_save_talk_data()
        train, dev, test = get_TDT_split(
            pd.read_csv(ATTACK_AGGRESSION_FN, index_col=0).fillna(' '))
        cnames = train.columns.values[0:2]
        aucfn = "auc_scores_attack.csv"
    elif myargs['-dataset'] == 'toxic':
        train, dev, test = get_TDT_split(
            pd.read_csv('train.csv').fillna(' '))
        cnames = CLASS_NAMES
        aucfn = "auc_scores.csv"
    if "-local" in myargs:
        local_path = myargs["-local"]
        if os.path.isfile(local_path):
            if "-dims" in myargs:
                dims = int(myargs["-dims"])
                use_local_embeddings = True
                path = local_path
            else:
                print "Please speficy the number of dimensions to use custom vectors"
        else:
            print "Could not find embeddings with that path"

    auc_scores = fit_logistic_ngram_sklearn(train, dev, test, dataset=myargs['-dataset'], cnames=cnames)
    save_auc_scores(auc_scores, APPROACH, CLASSIFIER, FLAVOR, cnames=cnames,
                    fn=aucfn)
    print('Avg ROC score is {}'.format(np.mean(auc_scores)))
