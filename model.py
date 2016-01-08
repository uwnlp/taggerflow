import collections
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import logging
import features

from custom_rnn_cell import *
from custom_rnn import *

class SupertaggerModel(object):
    lstm_hidden_size = 128
    penultimate_hidden_size = 64
    num_layers = 1

    # If variables in the computation graph are frozen, the protobuffer can be used out of the box.
    def __init__(self, config, data, is_training, freeze=False):
        self.config = config
        self.max_tokens = max_tokens = data.train_max_tokens if is_training else data.dev_max_tokens

        # Redeclare some variables for convenience.
        supertags_size = data.supertag_space.size()
        embedding_spaces = data.embedding_spaces

        with tf.name_scope("inputs"):
            # Each training step is batched with a maximum length.
            self.x = tf.placeholder(tf.int32, [max_tokens, None, len(embedding_spaces)], name="x")
            self.num_tokens = tf.placeholder(tf.int64, [None], name="num_tokens")
            if is_training:
                self.y = tf.placeholder(tf.int32, [max_tokens, None], name="y")
                self.tritrain = tf.placeholder(tf.float32, [None], name="tritrain")
                self.weights = tf.placeholder(tf.float32, [max_tokens, None], name="weights")

        # From feature indexes to concatenated embeddings.
        with tf.name_scope("embeddings"):
            with tf.device("/cpu:0"):
                embeddings_w = collections.OrderedDict((name, maybe_get_variable(name, [space.size(), space.embedding_size], freeze=freeze)) for name, space in embedding_spaces.items())
                embeddings = [tf.nn.embedding_lookup(e,i) for e,i in zip(embeddings_w.values(), tf.split(2, len(embedding_spaces), self.x))]
            concat_embedding = tf.concat(3, embeddings)
            concat_embedding = tf.squeeze(concat_embedding, [2])
            if is_training:
                concat_embedding = tf.nn.dropout(concat_embedding, 1.0 - config.dropout_probability)

        with tf.name_scope("lstm"):
            # LSTM cell is replicated across stacks and timesteps.
            first_cell = DyerLSTMCell(self.lstm_hidden_size, concat_embedding.get_shape()[2].value, freeze=freeze)
            if self.num_layers > 1:
                stacked_cell = DyerLSTMCell(self.lstm_hidden_size, self.lstm_hidden_size, freeze=freeze)
                cell = rnn_cell.MultiRNNCell([first_cell] + [stacked_cell] * (self.num_layers - 1))
            else:
                cell = first_cell

            # Split into LSTM inputs.
            inputs = tf.unpack(concat_embedding)

            # Construct LSTM.
            outputs = bidirectional_rnn(cell, cell, inputs, dtype=tf.float32, sequence_length=self.num_tokens)

            # Rejoin LSTM outputs.
            outputs = tf.pack(outputs)

        with tf.name_scope("softmax"):
            # From LSTM outputs to softmax.
            flattened = self.flatten(outputs)
            penultimate = tf.nn.relu(linear(flattened, self.penultimate_hidden_size, "penultimate", freeze=freeze))
            logits = linear(penultimate, supertags_size, "softmax", freeze=freeze)

        with tf.name_scope("prediction"):
            self.probabilities = self.unflatten(tf.nn.softmax(logits), name="probabilities")

        if is_training:
            with tf.name_scope("loss"):
                modified_weights = self.weights * tf.expand_dims(config.ccgbank_weight * (1.0 - self.tritrain) +  self.tritrain, 0)
                targets = self.flatten(self.y)

                self.loss = seq2seq.sequence_loss([logits],
                                                  [self.flatten(self.y)],
                                                  [self.flatten(modified_weights)],
                                                  supertags_size,
                                                  average_across_timesteps=True, average_across_batch=True)

                params = tf.trainable_variables()
                if self.config.regularization > 0.0:
                    self.regularization = self.config.regularization * sum(tf.nn.l2_loss(p) for p in params)
                else:
                    self.regularization = tf.constant(0.0)
                self.cost = self.loss + self.regularization

            # Construct training operations.
            with tf.name_scope("training"):
                self.global_step = tf.get_variable("global_step", [], trainable=False, initializer=tf.constant_initializer(0))
                optimizer = tf.train.AdamOptimizer()
                grads = tf.gradients(self.cost, params)
                grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)
                self.optimize = optimizer.apply_gradients(zip(grads, params), global_step=self.global_step)

    # Commonly used reshaping operations.
    def flatten(self, x):
        if len(x.get_shape()) == 2:
            return tf.reshape(x, [-1])
        elif len(x.get_shape()) == 3:
            return tf.reshape(x, [-1, x.get_shape()[2].value])
        else:
            raise ValueError("Unsupported shape: {}".format(x.get_shape()))

    def unflatten(self, flattened, name=None):
        return tf.reshape(flattened, [self.max_tokens, -1, flattened.get_shape()[1].value], name=name)
