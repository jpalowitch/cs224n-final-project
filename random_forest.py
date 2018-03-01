import numpy as np
import tensorflow as tf
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
import pandas as pd
import pickle
from project_utils import get_base2_labels, get_TDT_split, get_onehots_from_labels, CLASS_NAMES
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = ""
TRAIN_DATA_FILE = "train.csv"
TFIDF_VECTOR_FILE = "tdidf.pkl"
SENTENCE_VECTORS_FILE = "sentence_vectors.pkl"
train_data = pd.read_csv(TRAIN_DATA_FILE)

# parameters
NUM_FEATURES = 10000

# NOTE: implementation based off of
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/random_forest.py

class Config:
    """ Contains the hyper parameters and configuration for the random forest model.
    """
    num_classes = 2
    num_features = 10000
    num_steps = 20
    batch_size = 4112
    num_trees = 100
    max_nodes = 100

def vectorize_corpus_tf_idf(df, path=SENTENCE_VECTORS_FILE):
    """ Vectorizes the corpus using tf-idf.

    Args:
        df: input data, pandas data frame
        path: path to data file
    Returns:
        sentence_vectors: sentences vectorized using tf-idf
    """
    if os.path.isfile(path):
        # with open(TFIDF_VECTOR_FILE, "rb") as fp:
        #     feature_dict = pickle.load(fp)
        with open(path, "rb") as fp:
            sentence_vectors = pickle.load(fp)
        return sentence_vectors
    else:
        comments = df[['comment_text']].values.flatten()
        vectorizer = TfidfVectorizer(
            max_features=NUM_FEATURES,
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(1, 2)
        )

        X = vectorizer.fit_transform(comments)
        idf = vectorizer.idf_
        # feature_dict = dict(zip(vectorizer.get_feature_names(), idf))
        sentence_vectors = X.toarray()

        # save vectors and sentences
        # uncomment to save vectors
        # with open(TFIDF_VECTOR_FILE, "wb") as fp:
        #     pickle.dump(feature_dict, fp)
        with open(path, "wb") as fp:
            pickle.dump(sentence_vectors, fp)
        return sentence_vectors


class RandomForest():
    """Builds a random forest model for training.
    """
    def __init__(self, config):
        self.config = config
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.forest_graph = None

    def add_graph(self):
        """ Builds the forest graph based off of the hyper parameters in Config.
        """
        hyper_parameters = tensor_forest.ForestHParams(num_classes=self.config.num_classes,
                                                            num_features=self.config.num_features,
                                                            num_trees=self.config.num_trees,
                                                            max_nodes=self.config.max_nodes).fill()
        self.forest_graph = tensor_forest.RandomForestGraphs(hyper_parameters)

    def create_feed_dict(self, inputs, labels):
        """ Creates a dictionary to feed data into the model

        Returns:
            Feed dictionary
        """
        return {
            self.inputs_placeholder: inputs,
            self.labels_placeholder: labels
        }

    def add_placeholders(self):
        """Adds placeholder values to the model for reading batches of data
        """
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=[None, self.config.num_features])
        self.labels_placeholder = tf.placeholder(tf.int32, shape=[None])

    def add_training_op(self):
        """ Adds the training tensor.
        """
        self.train_op = self.forest_graph.training_graph(self.inputs_placeholder, self.labels_placeholder)

    def add_loss_op(self):
        """ Adds the loss tensor.
        """
        self.loss_op = self.forest_graph.training_loss(self.inputs_placeholder, self.labels_placeholder)

    def add_prediction_op(self):
        """ Adds tensor for predicting the class.
        """
        infer_op, _, _ = self.forest_graph.inference_graph(self.inputs_placeholder)
        # Uncomment below lines to use default accuracy metric
        # correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(self.labels_placeholder, tf.int64))
        # self.accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.infer_op = infer_op

    def train_on_batch(self, inputs_batch, labels_batch, sess):
        """Trains the model on the minibatch.

        Args:
            inputs_batch: feature vector batch
            labels_batch: labels batch
            sess: current TensorFlow session
        Returns:
            loss: the loss on batch.
        """
        feed_dict = self.create_feed_dict(inputs_batch, labels_batch)
        _, loss = sess.run([self.train_op, self.loss_op], feed_dict=feed_dict)
        return loss

    def do_train(self, sess, features, labels, targets):
        """Iterates over the number of epochs and trains the model by each minibatch.

        Args:
            sess: current TensorFlow session
            features: features vectors
            labels: vector of integer labels
            targets: row vectors of one-hot labels
        Returns:
            loss: final loss
        """
        loss = 0.0
        for i in range(self.config.num_steps):
            for inputs_batch, labels_batch, targets_batch in minibatch(features[:], labels[:], targets[:], self.config.batch_size):
                loss += self.train_on_batch(inputs_batch, labels_batch, sess)

            if i % 1 == 0 or i == 1:
                preds = sess.run(self.infer_op, feed_dict={self.inputs_placeholder: inputs_batch})
                auc = roc_auc_score(targets_batch, preds)
                print 'Epoch {}, Loss: {}, roc-auc: {}'.format(i, loss, auc)
        return loss

    def build(self):
        """Builds the model and returns the operators.
        """
        self.add_placeholders()
        self.add_graph()
        train_op = self.add_training_op()
        loss_op = self.add_loss_op()
        self.add_prediction_op()

    def do_test(self, sess, inputs, targets, labels):
        """Tests the model.

        Args:
            sess: current tensorflow session
            inputs: input vectors
            targets: correct labels
        """
        preds = sess.run(self.infer_op, feed_dict={self.inputs_placeholder: inputs})
        auc = roc_auc_score(targets, preds)
        print 'Test roc-auc: {}'.format(auc)


def train_and_test_model(train_features, train_labels, train_targets, test_features, test_targets, test_labels):
    """Trains and tests random forest model.

    Args:
        train_features: training features
        train_labels: training labels, integer values
        train_targets: one-hot row vector of labels
        test_features: testing features
        test_targets: one-hot row vector of test labels
        test_labels: labels for testing
    Returns:
        The loss for the model
    """
    # initialize model and build it
    config = Config()
    forest_model = RandomForest(config)
    forest_model.build()

    # initialize tensorflow variables
    init_vars = tf.group(tf.global_variables_initializer(),
                        resources.initialize_resources(resources.shared_resources()))

    with tf.Session() as session:
        session.run(init_vars)

        # train
        loss = forest_model.do_train(session, train_features, train_labels, train_targets)
        print "Final train loss: {}".format(loss)

        # test
        forest_model.do_test(session, test_features, test_targets, test_labels)
    return loss


def minibatch(inputs, labels, targets, batch_size, shuffle=True):
    """ Performs minibatching on set of data. Based off of stack overflow post:
    https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python

    Args:
        inputs: feature matrix
        labels: label vector
        batch_size: size of batch to sample
        shuffle: whether to randomly shuffle indices
    Returns:
        a batch of inputs and labels
    """
    assert inputs.shape[0] == labels.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for i in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        if shuffle:
            batch = indices[i:(i + batch_size)]
        else:
            batch = slice(i, i + batch_size)
        yield inputs[batch], labels[batch], targets[batch]

if __name__ == "__main__":
    train, dev, test = get_TDT_split(pd.read_csv('train.csv').fillna(' '))
    sentence_vectors = vectorize_corpus_tf_idf(train)
    dev_sentence_vectors = vectorize_corpus_tf_idf(dev, path="dev_sentence_vectors.pkl")
    test_sentence_vectors = vectorize_corpus_tf_idf(test, path="test_sentence_vectors.pkl")

    for target_class in range(6):
        with tf.Graph().as_default() as graph:
            print '---- training and testing: {} ----'.format(CLASS_NAMES[target_class])
            train_labels = train[CLASS_NAMES[target_class]].values
            train_target = get_onehots_from_labels(train_labels)
            dev_labels = dev[CLASS_NAMES[target_class]].values
            dev_target = get_onehots_from_labels(dev_labels)
            # test_labels = test[CLASS_NAMES[target_class]].values
            # test_target = get_onehots_from_labels(test_labels)
            # train
            train_and_test_model(sentence_vectors, train_labels, train_target,  \
                dev_sentence_vectors, dev_target, dev_labels)
