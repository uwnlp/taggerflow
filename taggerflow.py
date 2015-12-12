#!/usr/bin/env python

import sys
import json
import math
import argparse

from collections import defaultdict
from collections import OrderedDict
from collections import Counter

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import logging

import ccgbank
import util

logger = None

def initialize_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    # create file handler and set level to log
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

class SuperTaggerModel(object):

    def __init__(self, config):
        self.config = config

        # Redeclare some configuration settings for convenience.
        batch_size = config.batch_size
        supertags_size = config.supertag_space.size()
        embedding_spaces = config.embedding_spaces
        lstm_hidden_size = config.lstm_hidden_size
        relu_hidden_size = config.relu_hidden_size
        max_tokens = config.max_tokens

        # Each training step is batched with a maximum length.
        self.x = tf.placeholder(tf.int32, [batch_size, max_tokens, len(embedding_spaces)])
        self.y = tf.placeholder(tf.int32, [batch_size, max_tokens])
        self.num_tokens = tf.placeholder(tf.int64, [batch_size])
        self.keep_probability = tf.constant(config.keep_probability)

        # LSTM cell is replicated across stacks and timesteps.
        lstm_cell = rnn_cell.BasicLSTMCell(lstm_hidden_size, forget_bias=1.0)
        lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_probability)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        # Both LSTMs have their own initial state.
        initial_state_fw = cell.zero_state(batch_size, tf.float32)
        initial_state_bw = cell.zero_state(batch_size, tf.float32)

        # From feature indexes to concatenated embeddings.
        embedding_tables = [tf.get_variable("{}_embedding".format(name), [space.size(), space.embedding_size]) for name, space in embedding_spaces.items()]
        embeddings = [tf.squeeze(tf.nn.embedding_lookup(e,i), [2]) for e,i in zip(embedding_tables, tf.split(2, len(embedding_spaces), self.x))]
        concat_embedding = tf.concat(2, embeddings)

        # From concatenated embeddings to LSTM inputs.
        inputs = self.linear_layer("concat", concat_embedding, lstm_hidden_size)
        inputs = tf.split(1, max_tokens, inputs)
        inputs = [tf.squeeze(i, [1]) for i in inputs]

        # Construct LSTM.
        outputs = rnn.bidirectional_rnn(cell, cell, inputs,
                                        initial_state_fw=initial_state_fw,
                                        initial_state_bw=initial_state_bw,
                                        sequence_length=self.num_tokens)

        # Rejoin LSTM outputs.
        outputs = tf.concat(1, outputs)
        outputs = tf.reshape(outputs, [batch_size, max_tokens, -1])

        # From LSTM outputs to softmax.
        relu = tf.nn.relu(self.linear_layer("relu", outputs, 2 * relu_hidden_size))
        relu = tf.nn.dropout(relu, self.keep_probability)
        softmax = self.linear_layer("softmax", relu, supertags_size)

        # Predictions are the indexes with the highest value from the softmax layer.
        self.predictions = tf.argmax(softmax, 2)

        # Cross-entropy loss.
        pseudo_batch_size = batch_size * max_tokens
        self.loss = seq2seq.sequence_loss_by_example([tf.reshape(softmax, [pseudo_batch_size, -1])],
                                                     [tf.reshape(self.y, [pseudo_batch_size])],
                                                     [tf.ones([pseudo_batch_size])],
                                                     supertags_size)
        self.loss = tf.reduce_sum(self.loss) / batch_size

        # Construct training operation.
        params = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, params), config.max_grad_norm)
        self.optimize = tf.train.GradientDescentOptimizer(config.learning_rate).apply_gradients(zip(grads, params))

    # xs contains (batch, timestep, x)
    # Performs y = xw + b.
    # Returns the result containing (batch, timestep, x * w + b)
    def linear_layer(self, name, xs, y_dim):
        xs_dims = [d.value for d in xs.get_shape()]
        w = tf.get_variable("{}_w".format(name), [xs_dims[2], y_dim])
        b = tf.get_variable("{}_b".format(name), [y_dim])
        flattened_xs = tf.reshape(xs, [-1, xs_dims[2]])
        ys = tf.nn.xw_plus_b(flattened_xs, w, b)
        return tf.reshape(ys, [xs_dims[0], xs_dims[1], y_dim])

    def get_embedding_indexes(self, token):
        return [space.index(space.extract_from_token(token)) for space in self.config.embedding_spaces.values()]

    def get_batches(self, sentences):
        data = [([self.get_embedding_indexes(t) for t in tokens], [self.config.supertag_space.index(s) for s in supertags]) for tokens, supertags in sentences]

        batch_size = self.config.batch_size
        batches = []
        num_batches = int(math.ceil(len(data)/float(batch_size)))
        for i in range(num_batches):
            batch_x = np.zeros([batch_size, self.config.max_tokens, len(self.config.embedding_spaces)], dtype=np.int32)
            batch_y = np.zeros([batch_size, self.config.max_tokens], dtype=np.int32)
            batch_num_tokens = np.zeros([batch_size], dtype=np.int64)
            for j,(x,y) in enumerate(data[i * batch_size: (i + 1) * batch_size]):
                if len(x) > self.config.max_tokens:
                    logger.info("Skipping sentence of length {}.".format(len(x)))
                    continue
                batch_num_tokens[j] = len(x)
                batch_x[j,:len(x):] = x
                batch_y[j,:len(y)] = y
            batches.append((batch_x, batch_y, batch_num_tokens))
        return batches

    def evaluate(self, batches, session):
        num_correct = 0
        num_total = 0
        for x,y,num_tokens in batches:
            predictions = session.run(self.predictions, {
                self.x: x,
                self.num_tokens: num_tokens,
                self.keep_probability: 1.0
            })
            for i,n in enumerate(num_tokens):
                num_total += n
                num_correct += sum(int(predictions[i,j] == y[i,j]) for j in range(n))
        return (num_correct, num_total)

    def train(self, train_data, dev_data):
        logger.info("Massaging data into mini-batch format...")
        train_batches = self.get_batches(train_data)
        dev_batches = self.get_batches(dev_data)
        logger.info("Train batches: {}".format(len(train_batches)))
        logger.info("Dev batches: {}".format(len(dev_batches)))

        logger.info("Starting training for {} epochs.".format(self.config.num_epochs))
        with tf.Session() as session:
            tf.initialize_all_variables().run()
            for epoch in range(self.config.num_epochs):
                logger.info("========= Epoch {:02d} =========".format(epoch))
                num_correct, num_total = self.evaluate(dev_batches, session)
                logger.info("Validation accuracy: {:.3f}% ({}/{})".format((100.0 * num_correct)/num_total, num_correct, num_total))
                train_loss = 0.0
                for i,(x,y,num_tokens) in enumerate(train_batches):
                    _, loss = session.run([self.optimize, self.loss], {
                        self.x: x,
                        self.y: y,
                        self.num_tokens: num_tokens
                    })
                    train_loss += loss
                    if i % 100 == 0:
                        logger.info("{}/{} mean training loss: {:.3f}".format(i+1, len(train_batches), train_loss/(i+1)))
                logger.info("Epoch mean training loss: {:.3f}".format(train_loss/len(train_batches)))
                logger.info("============================")

class SuperTaggerConfig(object):

    def __init__(self, supertag_space, embedding_spaces, config_file):
        self.supertag_space = supertag_space
        self.embedding_spaces = embedding_spaces

        with open(config_file) as f:
            config = json.load(f)
            self.lstm_hidden_size = config["lstm_hidden_size"]
            self.relu_hidden_size = config["relu_hidden_size"]
            self.num_layers = config["num_layers"]
            self.max_grad_norm = config["max_grad_norm"]
            self.num_epochs = config["num_epochs"]
            self.learning_rate = config["learning_rate"]
            self.max_tokens = config["max_tokens"]
            self.batch_size = config["batch_size"]
            self.keep_probability = config["keep_probability"]
            for name,size in config["embedding_sizes"].items():
                self.embedding_spaces[name].embedding_size = size

class FeatureSpace(object):
    def __init__(self, sentences, min_count=None):
        counts = Counter(self.extract(sentences))
        self.space = [f for f in counts if min_count is None or counts[f] >= min_count]

        # Append extra index for unknown features.
        num_known_features = len(self.space)
        self.ispace = defaultdict(lambda:num_known_features, {f:i for i,f in enumerate(self.space)})
        self.space.append(None)

    def index(self, f):
        return self.ispace[f]

    def feature(self, i):
        return self.space[i]

    def size(self):
        return len(self.space)

    def extract(self, sentence):
        raise NotImplementedError("Subclasses must implement this!")

class SupertagSpace(FeatureSpace):
    def __init__(self, sentences, min_count=None):
        super(SupertagSpace, self).__init__(sentences, min_count)

    def extract(self, sentences):
        for tokens, supertags in sentences:
            for s in supertags:
                yield s

class EmbeddingSpace(FeatureSpace):
    def __init__(self, sentences, min_count=None):
        super(EmbeddingSpace, self).__init__(sentences, min_count)

        # To be set by the configuration.
        self.embedding_size = None

    def extract(self, sentences):
        for tokens, supertags in sentences:
            for t in tokens:
                yield self.extract_from_token(t)

    def extract_from_token(self, token):
        raise NotImplementedError("Subclasses must implement this!")

class PretrainedEmbeddingSpace(EmbeddingSpace):
    def __init__(self, embeddings_file):
        already_added = set()
        self.embedding_size = None
        self.space = []
        self.embeddings = []
        with open(embeddings_file) as f:
            for i,line in enumerate(f.readlines()):
                splits = line.split()
                word = splits[0].lower()

                if i == 0 and word != "*unknown*":
                    raise ValueError("First embedding in the file should represent the unknown word.")
                if word not in already_added:
                    embedding = [float(s) for s in splits[1:]]

                    if self.embedding_size is None:
                        self.embedding_size = len(embedding)
                    elif self.embedding_size != len(embedding):
                        raise ValueError("Dimensions mismatch. Expected {} but was {}.".format(self.embedding_size, len(embedding)))

                    already_added.add(word)
                    self.space.append(word)
                    self.embeddings.append(embedding)

        self.space = list(self.space)
        self.ispace = defaultdict(lambda:0, {f:i for i,f in enumerate(self.space)})

class WordSpace(PretrainedEmbeddingSpace):
    def __init__(self, embeddings_file):
        super(WordSpace, self).__init__(embeddings_file)

    def extract_from_token(self, token):
        return token.lower()

class PrefixSpace(EmbeddingSpace):
    def __init__(self, sentences, n, min_count=None):
        self.n = n
        super(PrefixSpace, self).__init__(sentences, min_count)

    def extract_from_token(self, token):
        return token[:self.n]

class SuffixSpace(EmbeddingSpace):
    def __init__(self, sentences, n, min_count=None):
        self.n = n
        super(SuffixSpace, self).__init__(sentences, min_count)

    def extract_from_token(self, token):
        return token[-self.n:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="configuration json file")
    parser.add_argument("-d", "--debug", help="uses a smaller training set for debugging", action="store_true")
    args = parser.parse_args()

    logger = initialize_logger("info.log")

    train_sentences, dev_sentences, test_sentences = ccgbank.CCGBankReader().get_splits(args.debug)
    logger.info("Train: {} | Dev: {} | Test: {}".format(len(train_sentences), len(dev_sentences), len(test_sentences)))


    supertag_space = SupertagSpace(train_sentences)
    embedding_spaces = OrderedDict([("words",    WordSpace(util.maybe_download("data",
                                                                               "http://appositive.cs.washington.edu/resources/",
                                                                               "embeddings.raw"))),
                                    ("prefix_1", PrefixSpace(train_sentences, 1, min_count=3)),
                                    ("prefix_2", PrefixSpace(train_sentences, 2, min_count=3)),
                                    ("prefix_3", PrefixSpace(train_sentences, 3, min_count=3)),
                                    ("prefix_4", PrefixSpace(train_sentences, 4, min_count=3)),
                                    ("suffix_1", SuffixSpace(train_sentences, 1, min_count=3)),
                                    ("suffix_2", SuffixSpace(train_sentences, 2, min_count=3)),
                                    ("suffix_3", SuffixSpace(train_sentences, 3, min_count=3)),
                                    ("suffix_4", SuffixSpace(train_sentences, 4, min_count=3))])

    logger.info("Number of supertags: {}".format(supertag_space.size()))
    for name, space in embedding_spaces.items():
        logger.info("Number of {}: {}".format(name, space.size()))

    config = SuperTaggerConfig(supertag_space, embedding_spaces, args.config)
    model = SuperTaggerModel(config)
    model.train(train_sentences, dev_sentences)
