import numpy as np
import tensorflow as tf
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
import pandas as pd
import pickle
from project_utils import get_base2_labels, get_TDT_split

os.environ["CUDA_VISIBLE_DEVICES"] = ""
TRAIN_DATA_FILE = "train.csv"
TFIDF_VECTOR_FILE = "tdidf.pkl"
SENTENCE_VECTORS_FILE = "sentence_vectors.pkl"
train_data = pd.read_csv(TRAIN_DATA_FILE)

# parameters
NUM_FEATURES = 1000

# NOTE: this requires Python 3, so make sure create a virtualenv with it
# NOTE: implementation based off of
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/random_forest.py

class Config:
    """ Contains the hyper parameters and configuration for the random forest model.
    """
    num_classes = 64
    num_features = 1000
    num_steps = 150
    batch_size = 1024
    num_trees = 10
    max_nodes = 1000

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
        """ Adds the training operator.

        Returns:
            train_op: tensor for training model
        """
        train_op = self.forest_graph.training_graph(self.inputs_placeholder, self.labels_placeholder)
        return train_op

    def add_loss_op(self):
        """ Adds the loss operator.

        Returns:
            loss_op: tensor for evaluating loss
        """
        loss_op = self.forest_graph.training_loss(self.inputs_placeholder, self.labels_placeholder)
        return loss_op

    def add_accuracy_op(self):
        """ Calculates the error and adds the accuracy operators.

        Returns:
            accuracy_op: tensor for evaluating model
        """
        infer_op, _, _ = self.forest_graph.inference_graph(self.inputs_placeholder)
        correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(self.labels_placeholder, tf.int64))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy_op

    def train_on_batch(self, inputs_batch, labels_batch, sess, train_op, loss_op):
        """Trains the model on the minibatch.

        Args:
            inputs_batch: feature vector batch
            labels_batch: labels batch
            sess: current TensorFlow session
            train_op: training operator
            loss_op: operator for calculating loss
        Returns:
            loss: the loss on batch.

        """
        feed_dict = self.create_feed_dict(inputs_batch, labels_batch)
        _, loss = sess.run([train_op, loss_op], feed_dict=feed_dict)
        return loss

    def do_train(self, train_op, loss_op, accuracy_op, sess, features, labels):
        """Iterates over the number of epochs and trains the model by each minibatch.

        Args:
            train_op: training operator
            loss_op: operator for calculating loss
            accuracy_op: operator for calculating accuracy
            sess: current TensorFlow session
            features: features vectors
            labels: labels
        Returns:
            loss: final loss
            accuracy_op: trained predictor
        """
        loss = 0.0
        for i in range(self.config.num_steps):
            for inputs_batch, labels_batch in minibatch(features[:], labels[:], self.config.batch_size):
                loss += self.train_on_batch(inputs_batch, labels_batch, sess, train_op, loss_op)

            if i % 10 == 0 or i == 1:
                feed_dict = self.create_feed_dict(inputs_batch, labels_batch)
                acc = sess.run(accuracy_op, feed_dict=feed_dict)
                print('Step %i, Loss: %f, Acc: %f' % (i, loss, acc))
        return loss, accuracy_op

    def build(self):
        """Builds the model and returns the operators.

        Returns:
            train_op: training operator
            loss_op: operator for calculating loss
            accuracy_op: operator for calculating accuracy
        """
        self.add_placeholders()
        self.add_graph()
        train_op = self.add_training_op()
        loss_op = self.add_loss_op()
        accuracy_op = self.add_accuracy_op()
        return [train_op, loss_op, accuracy_op]

    def do_test(self, test_vectors, test_labels, accuracy_op, session):
        """Tests the model once the accuracy operator has been trained.

        Args:
            test_vectors: test set feature vectors
            test_labels: test set labels
            accuracy_op: trained accracy operator
            session: current TensorFlow session
        """
        feed_dict = self.create_feed_dict(test_vectors, test_labels)
        print("Test Accuracy:", session.run(accuracy_op, feed_dict=feed_dict))


def train_and_test_model(train_features, train_labels, test_features, test_labels):
    """Trains and tests random forest model.

    Args:
        train_features: training features
        train_labels: training labels
        test_features: testing features
        test_labels: testing labels
    """
    # initialize model and build it
    config = Config()
    forest_model = RandomForest(config)
    operators = forest_model.build()

    # initialize tensorflow variables
    init_vars = tf.group(tf.global_variables_initializer(),
                        resources.initialize_resources(resources.shared_resources()))

    with tf.Session() as session:
        session.run(init_vars)
        # add args
        operators.append(session)
        operators.append(train_features)
        operators.append(train_labels)
        # train
        loss, accuracy_op = forest_model.do_train(*operators)
        print("Final train loss: {}".format(loss))
        # test
        forest_model.do_test(test_features, test_labels, accuracy_op, session)

if __name__ == "__main__":
    train, dev, test = get_TDT_split(pd.read_csv('train.csv').fillna(' '))
    # train
    sentence_vectors = vectorize_corpus_tf_idf(train)
    labels = get_base2_labels(train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values[:])
    # dev
    # dev_sentence_vectors = vectorize_corpus_tf_idf(dev, path="dev_sentence_vectors.pkl")
    # dev_labels = get_base2_labels(dev[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values[:])
    # test
    test_sentence_vectors = vectorize_corpus_tf_idf(test, path="test_sentence_vectors.pkl")
    test_labels = get_base2_labels(test[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values[:])
    # run it
    train_and_test_model(sentence_vectors, labels, test_sentence_vectors, test_labels)
