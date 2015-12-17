#!/usr/bin/env python

import os
import sys
import json
import math
import argparse
import collections
import logging

import numpy as np
import tensorflow as tf

import features
import model
import evaluation
import ccgbank
import util

class SupertaggerTask(object):

    def __init__(self, config_file, logdir, debug):
        self.logdir = logdir

        train_sentences, dev_sentences, test_sentences = ccgbank.CCGBankReader().get_splits(debug)
        logging.info("Train sentences: {}".format(len(train_sentences)))
        logging.info("Dev sentences: {}".format(len(dev_sentences)))

        supertag_space = features.SupertagSpace(train_sentences)
        embedding_spaces = collections.OrderedDict(
            [("words",    features.WordSpace(util.maybe_download("data",
                                                                 "http://appositive.cs.washington.edu/resources/",
                                                                 "embeddings.raw"), debug)),
             ("prefix_1", features.PrefixSpace(train_sentences, 1, min_count=3)),
             ("prefix_2", features.PrefixSpace(train_sentences, 2, min_count=3)),
             ("prefix_3", features.PrefixSpace(train_sentences, 3, min_count=3)),
             ("prefix_4", features.PrefixSpace(train_sentences, 4, min_count=3)),
             ("suffix_1", features.SuffixSpace(train_sentences, 1, min_count=3)),
             ("suffix_2", features.SuffixSpace(train_sentences, 2, min_count=3)),
             ("suffix_3", features.SuffixSpace(train_sentences, 3, min_count=3)),
             ("suffix_4", features.SuffixSpace(train_sentences, 4, min_count=3))])

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
            train_model = model.SupertaggerModel(self.config, self.config.batch_size)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            dev_model = model.SupertaggerModel(self.config, self.dev_batches[0][1].shape[0])

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

            with evaluation.SupertaggerEvaluationContext(session, self.dev_batches, dev_model, global_step, writer):
                for epoch in range(self.config.num_epochs):
                    logging.info("========= Epoch {:02d} =========".format(epoch))
                    train_loss = 0.0
                    for i,((x,num_tokens),y) in enumerate(self.train_batches):
                        _, loss = session.run([optimize, train_model.loss], {
                            train_model.x: x,
                            train_model.y: y,
                            train_model.num_tokens: num_tokens
                        })
                        train_loss += loss
                        if i % 10 == 0:
                            logging.info("{}/{} mean training loss: {:.3f}".format(i+1, len(self.train_batches), train_loss/(i+1)))

                    train_loss = train_loss / len(self.train_batches)
                    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Train Loss", simple_value=train_loss)]),
                                       tf.train.global_step(session, global_step))
                    logging.info("Epoch mean training loss: {:.3f}".format(train_loss))

                    timer.tick("{}/{} epochs".format(epoch + 1, self.config.num_epochs))
                    logging.info("============================")

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
