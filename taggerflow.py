#!/usr/bin/env python

import os
import sys
import json
import math
import argparse
import time

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

class SupertaggerModel(object):

    def __init__(self, config, batch_size):
        self.config = config

        # Redeclare some configuration settings for convenience.
        supertags_size = config.supertag_space.size()
        embedding_spaces = config.embedding_spaces
        max_tokens = config.max_tokens

        with tf.name_scope("inputs"):
            # Each training step is batched with a maximum length.
            self.x = tf.placeholder(tf.int32, [batch_size, max_tokens, len(embedding_spaces)], name="x")
            self.y = tf.placeholder(tf.int32, [batch_size, max_tokens], name="y")
            self.num_tokens = tf.placeholder(tf.int64, [batch_size], name="num_tokens")
            self.keep_probability = tf.constant(config.keep_probability, tf.float32, [], name="keep_probability")

        # From feature indexes to concatenated embeddings.
        with tf.name_scope("embeddings"), tf.device("/cpu:0"):
            self.embeddings_w = OrderedDict((name, tf.get_variable("{}_embedding_w".format(name), [space.size(), space.embedding_size])) for name, space in embedding_spaces.items() )
            embeddings = [tf.squeeze(tf.nn.embedding_lookup(e,i), [2]) for e,i in zip(self.embeddings_w.values(), tf.split(2, len(embedding_spaces), self.x))]
            concat_embedding = tf.concat(2, embeddings)
            concat_embedding = tf.nn.dropout(concat_embedding, self.keep_probability)

        with tf.name_scope("lstm"):
            # Split into LSTM inputs.
            inputs = tf.split(1, max_tokens, concat_embedding)
            inputs = [tf.squeeze(i, [1]) for i in inputs]

            cell_state_size = concat_embedding.get_shape()[2].value

            # LSTM cell is replicated across stacks and timesteps.
            lstm_cell = rnn_cell.BasicLSTMCell(cell_state_size)
            cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

            # Both LSTMs have their own initial state.
            initial_state_fw = tf.get_variable("initial_state_fw", [1, cell.state_size])
            initial_state_bw = tf.get_variable("initial_state_bw", [1, cell.state_size])

            # Construct LSTM.
            outputs = rnn.bidirectional_rnn(cell, cell, inputs,
                                            initial_state_fw=tf.tile(initial_state_fw, [batch_size, 1]),
                                            initial_state_bw=tf.tile(initial_state_bw, [batch_size, 1]),
                                            sequence_length=self.num_tokens)

            # Rejoin LSTM outputs.
            outputs = tf.concat(1, outputs)
            outputs = tf.reshape(outputs, [batch_size, max_tokens, -1])

        with tf.name_scope("softmax"):
            # From LSTM outputs to softmax.
            penultimate = tf.tanh(self.linear_layer("penultimate", outputs, config.penultimate_hidden_size))
            softmax = self.linear_layer("softmax", penultimate, supertags_size)

        with tf.name_scope("prediction"):
            # Predictions are the indexes with the highest value from the softmax layer.
            self.prediction = tf.argmax(softmax, 2)

        with tf.name_scope("loss"):
            # Cross-entropy loss.
            pseudo_batch_size = batch_size * max_tokens

            self.loss = seq2seq.sequence_loss([tf.reshape(softmax, [pseudo_batch_size, -1])],
                                              [tf.reshape(self.y, [pseudo_batch_size])],
                                              [tf.ones([pseudo_batch_size])],
                                              supertags_size,
                                              average_across_timesteps=False,
                                              average_across_batch=False)

            # Only average across valid tokens rather than padding.
            self.loss = self.loss / tf.cast(tf.reduce_sum(self.num_tokens), tf.float32)

            # Add L2 regularization for all trainable parameters.
            self.params = tf.trainable_variables()
            self.loss += 10e-6 * sum(tf.nn.l2_loss(p) for p in self.params)

        # Construct training operation.
        self.optimizer = tf.train.AdamOptimizer()

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

    def initialize(self, session):
        for name, space in self.config.embedding_spaces.items():
            if isinstance(space, PretrainedEmbeddingSpace):
                session.run(tf.assign(self.embeddings_w[name], space.embeddings))

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
    def __init__(self, embeddings_file, debug=False):
        already_added = set()
        self.embedding_size = None
        self.space = []
        self.embeddings = []
        with open(embeddings_file) as f:
            for i,line in enumerate(f.readlines()):
                if debug and i > 10:
                    break

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
    def __init__(self, embeddings_file, debug=False):
        super(WordSpace, self).__init__(embeddings_file, debug)

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

class SupertaggerEvaluationContext(util.ThreadedContext):
    def __init__(self, session, data, model, global_step, writer):
        super(SupertaggerEvaluationContext, self).__init__()
        self.session = session
        self.data = data
        self.model = model
        self.global_step = global_step
        self.writer = writer

    def loop(self):
        time.sleep(120)
        with util.Timer("Dev evaluation"):
            num_correct = 0
            num_total = 0
            for (x,num_tokens),y in self.data:
                prediction = self.session.run(self.model.prediction, {
                    self.model.x: x,
                    self.model.num_tokens: num_tokens,
                    self.model.keep_probability: 1.0
                })
                for i,n in enumerate(num_tokens):
                    num_total += n
                    num_correct += sum(int(prediction[i,j] == y[i,j]) for j in range(n))

            accuracy = (100.0 * num_correct)/num_total
            self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Dev Accuracy", simple_value=accuracy)]),
                                    tf.train.global_step(self.session, self.global_step))
            logging.info("Dev accuracy: {:.3f}% ({}/{})".format(accuracy, num_correct, num_total))

class SupertaggerTask(object):

    def __init__(self, config_file, logdir, debug):
        self.logdir = logdir

        train_sentences, dev_sentences, test_sentences = ccgbank.CCGBankReader().get_splits(debug)
        logging.info("Train sentences: {}".format(len(train_sentences)))
        logging.info("Dev sentences: {}".format(len(dev_sentences)))


        supertag_space = SupertagSpace(train_sentences)
        embedding_spaces = OrderedDict([("words",    WordSpace(util.maybe_download("data",
                                                                                   "http://appositive.cs.washington.edu/resources/",
                                                                                   "embeddings.raw"), debug)),
                                        ("prefix_1", PrefixSpace(train_sentences, 1, min_count=3)),
                                        ("prefix_2", PrefixSpace(train_sentences, 2, min_count=3)),
                                        ("prefix_3", PrefixSpace(train_sentences, 3, min_count=3)),
                                        ("prefix_4", PrefixSpace(train_sentences, 4, min_count=3)),
                                        ("suffix_1", SuffixSpace(train_sentences, 1, min_count=3)),
                                        ("suffix_2", SuffixSpace(train_sentences, 2, min_count=3)),
                                        ("suffix_3", SuffixSpace(train_sentences, 3, min_count=3)),
                                        ("suffix_4", SuffixSpace(train_sentences, 4, min_count=3))])

        logging.info("Number of supertags: {}".format(supertag_space.size()))
        for name, space in embedding_spaces.items():
            logging.info("Number of {}: {}".format(name, space.size()))

        self.config = SupertaggerConfig(supertag_space, embedding_spaces, config_file)

        logging.info("Massaging data into mini-batch format...")

        self.train_batches = self.get_batches(train_sentences, self.config.batch_size)
        self.dev_batches = self.get_batches(dev_sentences, len(dev_sentences))

        logging.info("Train batches: {}".format(len(self.train_batches)))
        logging.info("Dev batches: {}".format(len(self.dev_batches)))

    def get_embedding_indexes(self, token):
        return [space.index(space.extract_from_token(token)) for space in self.config.embedding_spaces.values()]

    def get_batches(self, sentences, batch_size):
        data = [([self.get_embedding_indexes(t) for t in tokens], [self.config.supertag_space.index(s) for s in supertags]) for tokens, supertags in sentences]

        batches = []
        num_batches = int(math.ceil(len(data)/float(batch_size)))
        for i in range(num_batches):
            batch_x = np.zeros([batch_size, self.config.max_tokens, len(self.config.embedding_spaces)], dtype=np.int32)
            batch_y = np.zeros([batch_size, self.config.max_tokens], dtype=np.int32)
            batch_num_tokens = np.zeros([batch_size], dtype=np.int64)
            for j,(x,y) in enumerate(data[i * batch_size: (i + 1) * batch_size]):
                if len(x) > self.config.max_tokens:
                    logging.info("Skipping sentence of length {}.".format(len(x)))
                    continue
                batch_num_tokens[j] = len(x)
                batch_x[j,:len(x):] = x
                batch_y[j,:len(y)] = y
            batches.append(([batch_x, batch_num_tokens], batch_y))
        return batches

    def train(self, run_name):
        with tf.name_scope("initialization"):
            initializer = tf.random_uniform_initializer(-self.config.init_scale,
                                                        self.config.init_scale, seed=self.config.seed)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            train_model = SupertaggerModel(self.config, self.config.batch_size)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            dev_model = SupertaggerModel(self.config, self.get_validation_data()[0][1].shape[0])

        with tf.name_scope("training"):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            grads, _ = tf.clip_by_global_norm(tf.gradients(train_model.loss, train_model.params), self.config.max_grad_norm)
            optimize = train_model.optimizer.apply_gradients(zip(grads, train_model.params), global_step=global_step)

        with tf.Session() as session, util.Timer("Training") as timer:
            writer = tf.train.SummaryWriter(os.path.join(self.logdir, run_name), graph_def=session.graph_def, flush_secs=60)

            tf.initialize_all_variables().run()

            with util.Timer("Initializing model"):
                train_model.initialize(session)

            logging.info("Starting training for {} epochs.".format(self.config.num_epochs))

            with SupertaggerEvaluationContext(session, self.get_validation_data(), dev_model, global_step, writer):
                for epoch in range(self.config.num_epochs):
                    logging.info("========= Epoch {:02d} =========".format(epoch))
                    train_loss = 0.0
                    for i,((x,num_tokens),y) in enumerate(self.get_train_data()):
                        _, loss = session.run([optimize, train_model.loss], {
                            train_model.x: x,
                            train_model.y: y,
                            train_model.num_tokens: num_tokens
                        })
                        train_loss += loss
                        if i % 10 == 0:
                            logging.info("{}/{} mean training loss: {:.3f}".format(i+1, len(self.get_train_data()), train_loss/(i+1)))

                    train_loss = train_loss / len(self.get_train_data())
                    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Train Loss", simple_value=train_loss)]),
                                       tf.train.global_step(session, global_step))
                    logging.info("Epoch mean training loss: {:.3f}".format(train_loss))

                    timer.tick("{}/{} epochs".format(epoch + 1, self.config.num_epochs))
                    logging.info("============================")

    def get_train_data(self):
        return self.train_batches

    def get_validation_data(self):
        return self.dev_batches

    def get_test_data(self):
        return self.dev_batches

class SupertaggerConfig(object):

    def __init__(self, supertag_space, embedding_spaces, config_file):
        self.supertag_space = supertag_space
        self.embedding_spaces = embedding_spaces

        with open(config_file) as f:
            config = json.load(f)
            self.init_scale = config["init_scale"]
            self.seed = config["seed"]
            self.penultimate_hidden_size = config["penultimate_hidden_size"]
            self.num_layers = config["num_layers"]
            self.max_grad_norm = config["max_grad_norm"]
            self.num_epochs = config["num_epochs"]
            self.learning_rate = config["learning_rate"]
            self.momentum = config["momentum"]
            self.max_tokens = config["max_tokens"]
            self.batch_size = config["batch_size"]
            self.keep_probability = config["keep_probability"]
            for name,size in config["embedding_sizes"].items():
                if self.embedding_spaces[name].embedding_size is None:
                    self.embedding_spaces[name].embedding_size = size

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="configuration json file")
    parser.add_argument("-d", "--debug", help="uses a smaller training set for debugging", action="store_true")
    parser.add_argument("-r", "--run_name", help="named used to identify logs", default="default")
    parser.add_argument("-g", "--gpu", help="specify gpu devices to use")
    parser.add_argument("-l", "--logdir", help="directory to contain logs", default="logs")
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, args.run_name + ".log"), level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    task = SupertaggerTask(args.config, args.logdir, args.debug)
    task.train(args.run_name)
