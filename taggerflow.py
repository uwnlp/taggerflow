#!/usr/bin/env python

import sys
import json
import math
import argparse

from collections import defaultdict

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import ccgbank

class SuperTaggerModel(object):

    def __init__(self, config):
        self.config = config

        batch_size = config.batch_size
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        supertags_size = config.supertags_size
        max_tokens = config.max_tokens

        self.x = tf.placeholder(tf.int32, [batch_size, max_tokens])
        self.y = tf.placeholder(tf.int32, [batch_size, max_tokens])
        self.num_tokens = tf.placeholder(tf.int64, [batch_size])

        lstm_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        initial_state_fw = cell.zero_state(batch_size, tf.float32)
        initial_state_bw = cell.zero_state(batch_size, tf.float32)

        embedding = tf.get_variable("embedding", [vocab_size, hidden_size])
        inputs = tf.split(1, max_tokens, tf.nn.embedding_lookup(embedding, self.x))
        inputs = [tf.squeeze(i, [1]) for i in inputs]

        outputs = rnn.bidirectional_rnn(cell, cell, inputs,
                                        initial_state_fw=initial_state_fw,
                                        initial_state_bw=initial_state_bw,
                                        sequence_length=self.num_tokens)

        output = tf.reshape(tf.concat(1, outputs), [-1, 2 * hidden_size])
        softmax_w = tf.get_variable("softmax_w", [2 * hidden_size, supertags_size])
        softmax_b = tf.get_variable("softmax_b", [supertags_size])
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        self.predictions = tf.argmax(tf.reshape(logits, [batch_size, max_tokens, supertags_size]), 2)

        self.loss = seq2seq.sequence_loss_by_example([logits],
                                                     [tf.reshape(self.y, [-1])],
                                                     [tf.ones([batch_size * max_tokens])],
                                                     supertags_size)
        self.loss = tf.reduce_sum(self.loss) / batch_size

        params = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, params), config.max_grad_norm)
        self.optimize = tf.train.GradientDescentOptimizer(config.learning_rate).apply_gradients(zip(grads, params))

    def get_batches(self, data):
        batch_size = self.config.batch_size
        batches = []
        num_batches = int(math.ceil(len(data)/batch_size))
        for i in range(num_batches):
            batch_x = np.zeros([batch_size, self.config.max_tokens], dtype=np.int32)
            batch_y = np.zeros([batch_size, self.config.max_tokens], dtype=np.int32)
            batch_num_tokens = np.zeros([batch_size], dtype=np.int64)
            for j,(x,y) in enumerate(data[i * batch_size: (i + 1) * batch_size]):
                if len(x) > self.config.max_tokens:
                    continue
                batch_num_tokens[j] = len(x)
                batch_x[j,:len(x)] = x
                batch_y[j,:len(y)] = y
            batches.append((batch_x, batch_y, batch_num_tokens))
        return batches

    def evaluate(self, batches, session):
        num_correct = 0
        num_total = 0
        for x,y,num_tokens in batches:
            predictions = session.run(self.predictions, {
                self.x: x,
                self.num_tokens: num_tokens
            })
            for i,n in enumerate(num_tokens):
                num_total += n
                num_correct += sum(int(predictions[i,j] == y[i,j]) for j in range(n))
        return (num_correct, num_total)

    def train(self, train_data, dev_data):
        print("Massaging data into mini-batch format...")
        train_batches = self.get_batches(train_data)
        dev_batches = self.get_batches(dev_data)

        print("Starting training for {} epochs.".format(self.config.num_epochs))
        with tf.Session() as session:
            tf.initialize_all_variables().run()
            for epoch in range(self.config.num_epochs):
                print("========= Epoch {:02d} =========".format(epoch))
                num_correct, num_total = self.evaluate(dev_batches, session)
                print("Validation accuracy: {:.3f}% ({}/{})".format((100.0 * num_correct)/num_total, num_correct, num_total))

                print("Training mini-batches: {}".format(len(train_batches)))
                train_loss = 0.0
                for x,y,num_tokens in train_batches:
                    _, loss = session.run([self.optimize, self.loss], {
                        self.x: x,
                        self.y: y,
                        self.num_tokens: num_tokens
                    })
                    train_loss += loss
                print("Epoch training loss: {:.3f}".format(train_loss/len(train_batches)))
                print("============================")

class SuperTaggerConfig(object):

    def __init__(self, vocab_size, supertags_size, config_file):
        self.vocab_size = vocab_size
        self.supertags_size = supertags_size

        with open(config_file) as f:
            config = json.load(f)
            self.hidden_size = config["hidden_size"]
            self.num_layers = config["num_layers"]
            self.max_grad_norm = config["max_grad_norm"]
            self.num_epochs = config["num_epochs"]
            self.learning_rate = config["learning_rate"]
            self.max_tokens = config["max_tokens"]
            self.batch_size = config["batch_size"]

def get_io_spaces(sentences):
    token_space = set()
    supertag_space = set()
    for tokens, supertags in sentences:
        token_space.update(tokens)
        supertag_space.update(supertags)
    return list(token_space), list(supertag_space)

def encode_data(sentences, itoken_space, isupertag_space):
    return [([itoken_space[t] for t in tokens], [isupertag_space[s] for s in supertags]) for tokens, supertags in sentences]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="configuration json file")
    parser.add_argument("-d", "--debug", help="uses a smaller training set for debugging", action="store_true")
    args = parser.parse_args()

    train_sentences, dev_sentences, test_sentences = ccgbank.CCGBankReader().get_splits(args.debug)
    print("Train: {} | Dev: {} | Test: {}".format(len(train_sentences), len(dev_sentences), len(test_sentences)))

    token_space, supertag_space = get_io_spaces(train_sentences)

    # Append extra index for unknown words.
    num_known_words = len(token_space)
    itoken_space = defaultdict(lambda:num_known_words, {t:i for i,t in enumerate(token_space)})
    token_space.append("<unk>")

    isupertag_space = defaultdict(lambda:-1, {s:i for i,s in enumerate(supertag_space)})

    print("Vocab size: {} | Number of supertags: {}".format(len(token_space), len(supertag_space)))

    train_data = encode_data(train_sentences, itoken_space, isupertag_space)
    dev_data = encode_data(dev_sentences, itoken_space, isupertag_space)

    config = SuperTaggerConfig(len(token_space), len(supertag_space), args.config)
    model = SuperTaggerModel(config)
    model.train(train_data, dev_data)
