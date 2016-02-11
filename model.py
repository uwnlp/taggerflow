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
    def __init__(self, config, data, is_training, freeze=False, max_tokens=None):
        self.config = config
        self.max_tokens = max_tokens or data.max_tokens

        # Redeclare some variables for convenience.
        supertags_size = data.supertag_space.size()
        embedding_spaces = data.embedding_spaces

        with tf.name_scope("inputs"):
            if is_training:
                x_prop = (tf.int32, [self.max_tokens, len(embedding_spaces)])
                y_prop = (tf.int32, [self.max_tokens])
                num_tokens_prop = (tf.int64, [])
                tritrain_prop = (tf.float32, [])
                weights_prop = (tf.float32, [self.max_tokens])
                dtypes, shapes = zip(x_prop, y_prop, num_tokens_prop, tritrain_prop, weights_prop)
                input_queue = tf.RandomShuffleQueue(len(data.train_sentences) + len(data.tritrain_sentences), 0, dtypes, shapes=shapes)
                self.inputs = [tf.placeholder(dtype, shape) for dtype, shape in zip(dtypes, shapes)]
                self.input_enqueue = input_queue.enqueue(self.inputs)
                self.x, self.y, self.num_tokens, self.tritrain, self.weights = input_queue.dequeue_many(data.batch_size)
                self.requeue = input_queue.enqueue_many([self.x, self.y, self.num_tokens, self.tritrain, self.weights])
            else:
                # Each training step is batched with a maximum length.
                self.x = tf.placeholder(tf.int32, [None, self.max_tokens, len(embedding_spaces)], name="x")
                self.num_tokens = tf.placeholder(tf.int64, [None], name="num_tokens")

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
            inputs = [tf.squeeze(split, [1]) for split in tf.split(1, self.max_tokens, concat_embedding)]

            # Construct LSTM.
            outputs = bidirectional_rnn(cell, cell, inputs, dtype=tf.float32, sequence_length=self.num_tokens)

            # Rejoin LSTM outputs.
            outputs = tf.concat(1, [tf.expand_dims(output, 1) for output in outputs])

        with tf.name_scope("softmax"):
            # From LSTM outputs to logits.
            flattened = self.flatten(outputs)
            penultimate = tf.nn.relu(linear(flattened, self.penultimate_hidden_size, "penultimate", freeze=freeze))
            logits = linear(penultimate, supertags_size, "softmax", freeze=freeze)

        with tf.name_scope("prediction"):
            self.probabilities = self.unflatten(tf.nn.softmax(logits), name="probabilities")

        if is_training:
            with tf.name_scope("loss"):
                modified_weights = self.weights * tf.expand_dims((1.0 - self.tritrain) +  config.tritrain_weight * self.tritrain, 1)

                """
                softmax = tf.nn.softmax(logits)
                softmax_list = [tf.squeeze(split, [1]) for split in tf.split(1, self.max_tokens, self.unflatten(softmax))]
                y_list = [tf.squeeze(split, [1]) for split in tf.split(1, self.max_tokens, self.y)]
                modified_weights_list = [tf.squeeze(split, [1]) for split in tf.split(1, self.max_tokens, modified_weights)]
                cross_entropy_list = [-tf.log(tf.gather(tf.transpose(s), y)) for s, y in zip(softmax_list, y_list)]
                cross_entropy_list = [tf.reduce_sum(ce * w) for ce, w in zip(cross_entropy_list, modified_weights_list)]
                self.loss = sum(cross_entropy_list)
                """
                self.loss = seq2seq.sequence_loss([logits],
                                                  [self.flatten(self.y)],
                                                  [self.flatten(modified_weights)],
                                                  average_across_timesteps=False, average_across_batch=False)

                params = tf.trainable_variables()

            # Construct training operations.
            with tf.name_scope("training"):
                self.global_step = tf.get_variable("global_step", [], trainable=False, initializer=tf.constant_initializer(0))
                optimizer = tf.train.MomentumOptimizer(0.01, 0.7)
                grads = tf.gradients(self.loss, params)
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
        return tf.reshape(flattened, [-1, self.max_tokens, flattened.get_shape()[1].value], name=name)
