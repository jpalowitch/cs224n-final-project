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

class Config:
    """ Contains the hyper parameters and configuration for the random forest model
    """
    num_classes = 64
    num_features = 1000
    num_steps = 1000000
    batch_size = 1024
    num_trees = 10
    max_nodes = 1000


def vectorize_corpus_tf_idf(df, path=SENTENCE_VECTORS_FILE):
    """ vectorizes the corpus using tf-idf
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
    def __init__(self, config):
        self.config = config
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.forest_graph = None

    def add_graph(self):
        hyper_parameters = tensor_forest.ForestHParams(num_classes=self.config.num_classes,
                                                            num_features=self.config.num_features,
                                                            num_trees=self.config.num_trees,
                                                            max_nodes=self.config.max_nodes).fill()
        self.forest_graph = tensor_forest.RandomForestGraphs(hyper_parameters)

    def create_feed_dict(self, inputs, labels):
        return {
            self.inputs_placeholder: inputs,
            self.labels_placeholder: labels
        }

    def add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=[None, self.config.num_features])
        self.labels_placeholder = tf.placeholder(tf.int32, shape=[None])

    def add_training_op(self):
        train_op = self.forest_graph.training_graph(self.inputs_placeholder, self.labels_placeholder)
        return train_op

    def add_loss_op(self):
        loss_op = self.forest_graph.training_loss(self.inputs_placeholder, self.labels_placeholder)
        return loss_op

    def add_accuracy_op(self):
        infer_op, _, _ = self.forest_graph.inference_graph(self.inputs_placeholder)
        correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(self.labels_placeholder, tf.int64))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy_op, correct_prediction

    def train_on_batch(self, inputs_batch, labels_batch, sess, train_op, loss_op):
        feed_dict = self.create_feed_dict(inputs_batch, labels_batch)
        _, loss = sess.run([train_op, loss_op], feed_dict=feed_dict)
        return loss

    def do_train(self, train_op, loss_op, accuracy_op, correct_prediction, sess, sentence_vectors, labels):
        loss = 0.0
        #TODO: implement actual batching...
        for i in range(0, self.config.num_steps - self.config.batch_size, self.config.batch_size):

            # need to transpose the sentence vector into row vector
            sentence_transpose = sentence_vectors[i:(i + self.config.batch_size)]
            label = labels[i:(i + self.config.batch_size)]
            loss += self.train_on_batch(sentence_transpose, label, sess, train_op, loss_op)

            if i % 50 == 0 or i == 1:
                feed_dict = self.create_feed_dict(sentence_transpose, label)
                acc = sess.run(accuracy_op, feed_dict=feed_dict)
                print('accuracy: {}'.format(acc))
                # print('correct_prediction: {}'.format(sess.run(correct_prediction)))
                print('Step %i, Loss: %f, Acc: %f' % (i, loss, acc))
        return loss, accuracy_op

    def build(self):
        self.add_placeholders()
        self.add_graph()
        train_op = self.add_training_op()
        loss_op = self.add_loss_op()
        accuracy_op, correct_prediction = self.add_accuracy_op()
        return [train_op, loss_op, accuracy_op, correct_prediction]

    def do_test(self, test_vectors, test_labels, accuracy_op, session):
        feed_dict = self.create_feed_dict(test_vectors, test_labels)
        print("Test Accuracy:", session.run(accuracy_op, feed_dict=feed_dict)) # 0.8969104


def train_model(sentence_vectors, labels, dev):
    # initialize model and build it
    config = Config()
    forest_model = RandomForest(config)
    operators = forest_model.build()

    # initialize tensorflow variables
    init_vars = tf.group(tf.global_variables_initializer(),
                        resources.initialize_resources(resources.shared_resources()))

    with tf.Session() as session:
    # session = tf.Session()
        session.run(init_vars)

        # add args
        operators.append(session)
        operators.append(sentence_vectors)
        operators.append(labels)
        # train
        loss, accuracy_op = forest_model.do_train(*operators)
        print('final train loss: {}'.format(loss))

        # test
        sentence_vectors_dev = vectorize_corpus_tf_idf(dev, path="dev_sentence_vectors.pkl")
        labels = get_base2_labels(dev[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values[:])
        forest_model.do_test(sentence_vectors_dev, labels, accuracy_op, session)

if __name__ == "__main__":
    train, dev, test = get_TDT_split(pd.read_csv('train.csv').fillna(' '))
    # train
    sentence_vectors = vectorize_corpus_tf_idf(train)
    labels = get_base2_labels(train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values[:])
    train_model(sentence_vectors, labels, dev)
