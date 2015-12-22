import collections

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import logging
import features
import custom_rnn_cell

class SupertaggerModel(object):

    def __init__(self, config, data):
        self.config = config

        # Redeclare some variables for convenience.
        batch_size = data.batch_size
        supertags_size = data.supertag_space.size()
        embedding_spaces = data.embedding_spaces
        max_tokens = data.max_tokens

        with tf.name_scope("inputs"):
            # Each training step is batched with a maximum length.
            self.x = tf.placeholder(tf.int32, [batch_size, max_tokens, len(embedding_spaces)], name="x")
            self.y = tf.placeholder(tf.int32, [batch_size, max_tokens], name="y")
            self.num_tokens = tf.placeholder(tf.int64, [batch_size], name="num_tokens")
            self.mask = tf.placeholder(tf.float32, [batch_size, max_tokens], name="mask")
            self.input_dropout_probability = tf.placeholder(tf.float32, [], name="dropout_probability")
            self.dropout_probability = tf.placeholder(tf.float32, [], name="dropout_probability")

        # From feature indexes to concatenated embeddings.
        with tf.name_scope("embeddings"), tf.device("/cpu:0"):
            embeddings_w = collections.OrderedDict((name, tf.get_variable("{}_embedding_w".format(name), [space.size(), space.embedding_size])) for name, space in embedding_spaces.items())
            embeddings = [tf.squeeze(tf.nn.embedding_lookup(e,i), [2]) for e,i in zip(embeddings_w.values(), tf.split(2, len(embedding_spaces), self.x))]
            concat_embedding = tf.concat(2, embeddings)
            concat_embedding = tf.nn.dropout(concat_embedding, 1.0 - self.input_dropout_probability)

        with tf.name_scope("lstm"):
            # LSTM cell is replicated across stacks and timesteps.
            first_cell = custom_rnn_cell.DyerLSTMCell(config.lstm_hidden_size, concat_embedding.get_shape()[2].value)
            stacked_cell = custom_rnn_cell.DyerLSTMCell(config.lstm_hidden_size, config.lstm_hidden_size)
            if config.num_layers > 1:
                cell = rnn_cell.MultiRNNCell([first_cell] + [stacked_cell] * (config.num_layers - 1))
            else:
                cell = first_cell
            cell = rnn_cell.DropoutWrapper(cell, output_keep_prob= 1.0 - self.dropout_probability)

            # Split into LSTM inputs.
            inputs = tf.split(1, max_tokens, concat_embedding)
            inputs = [tf.squeeze(i, [1]) for i in inputs]

            # Both LSTMs have their own initial state.
            initial_state_fw = tf.get_variable("initial_state_fw", [1, cell.state_size])
            initial_state_bw = tf.get_variable("initial_state_bw", [1, cell.state_size])

            # Construct LSTM.
            outputs = rnn.bidirectional_rnn(cell, cell, inputs,
                                            initial_state_fw=tf.tile(initial_state_fw, [batch_size, 1]),
                                            initial_state_bw=tf.tile(initial_state_bw, [batch_size, 1]),
                                            sequence_length=self.num_tokens)

            # Rejoin LSTM outputs.
            outputs = [tf.expand_dims(output, 1) for output in outputs]
            outputs = tf.concat(1, outputs)

        with tf.name_scope("softmax"):
            # From LSTM outputs to softmax.
            flattened = self.flatten(outputs, batch_size, max_tokens)
            penultimate = rnn_cell.linear(flattened, config.penultimate_hidden_size, True, scope="penultimate")
            name_to_nonlinearity = {
                "tanh" : tf.tanh,
                "relu" : tf.nn.relu,
                "relu6" : tf.nn.relu6
            }

            if config.penultimate_nonlinearity in name_to_nonlinearity:
                penultimate = name_to_nonlinearity[config.penultimate_nonlinearity](penultimate)
            else:
                raise ValueError("Unknown nonlinearity: {}".format(config.penultimate_nonlinearity))
            penultimate = tf.nn.dropout(penultimate, 1.0 - self.dropout_probability)
            softmax = rnn_cell.linear(penultimate, supertags_size, True, scope="softmax")

        with tf.name_scope("prediction"):
            # Predictions are the indexes with the highest value from the softmax layer.
            self.prediction = tf.argmax(self.unflatten(softmax, batch_size, max_tokens), 2)

        with tf.name_scope("loss"):
            # Cross-entropy loss.
            self.loss = seq2seq.sequence_loss([softmax],
                                              [self.flatten(self.y, batch_size, max_tokens)],
                                              [self.flatten(self.mask, batch_size, max_tokens)],
                                              supertags_size,
                                              average_across_timesteps=False)

            params = tf.trainable_variables()
            self.regularization = self.config.regularization * sum(tf.nn.l2_loss(p) for p in params)
            self.cost = self.loss + self.regularization

        # Construct training operations.
        with tf.name_scope("training"):
            optimizer = tf.train.AdamOptimizer()
            grads = tf.gradients(self.cost, params)
            grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.optimize = optimizer.apply_gradients(zip(grads, params), global_step=self.global_step)

        with tf.name_scope("initialization"):
            self.initializer = tf.random_uniform_initializer(-self.config.init_scale,
                                                             self.config.init_scale, seed=self.config.seed)
            self.initialize = tf.tuple(
                [tf.assign(embeddings_w[name], space.embeddings) for name,space in data.embedding_spaces.items()
                 if isinstance(space, features.PretrainedEmbeddingSpace)])

    # Commonly used reshaping operations.
    def flatten(self, x, batch_size, timesteps):
        if len(x.get_shape()) == 2:
            return tf.reshape(x, [batch_size * timesteps])
        elif len(x.get_shape()) == 3:
            return tf.reshape(x, [batch_size * timesteps, x.get_shape()[2].value])
        else:
            raise ValueError("Unsupported shape: {}".format(x.get_shape()))

    def unflatten(self, flattened, batch_size, timesteps):
        return tf.reshape(flattened, [batch_size, timesteps, -1])
