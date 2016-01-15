#!/usr/bin/env python

import math
import collections
import logging

import numpy as np

from features import *
from ccgbank import *
from util import *

class SupertaggerData(object):
    train_max_tokens = 102
    dev_max_tokens = 102
    batch_size = 4
    bucket_size = 5
    max_tritrain_length = 72

    def __init__(self, supertag_space, embedding_spaces, train_sentences, tritrain_sentences, dev_sentences):
        self.supertag_space = supertag_space
        self.embedding_spaces = embedding_spaces

        logging.info("Number of supertags: {}".format(self.supertag_space.size()))
        for name, space in self.embedding_spaces.items():
            logging.info("Number of {}: {}".format(name, space.size()))

        logging.info("Train sentences: {}".format(len(train_sentences)))
        logging.info("Tri-train sentences: {}".format(len(tritrain_sentences)))
        logging.info("Dev sentences: {}".format(len(dev_sentences)))

        if len(tritrain_sentences) > 0:
            train_distribution = self.get_sentence_length_distribution(train_sentences)
            tritrain_distribution = self.get_sentence_length_distribution(tritrain_sentences)

            logging.info("Train distribution: {}".format(self.format_distribution(train_distribution)))
            logging.info("Tri-train distribution: {}".format(self.format_distribution(tritrain_distribution)))

            self.distribution_ratios = [t/tt if tt > 0.0 else 0.0 for t,tt in zip(train_distribution, tritrain_distribution)]
            self.tritrain_ratio = float(len(train_sentences))/len(tritrain_sentences)

            logging.info("Distribution ratios: {}".format(self.format_distribution(self.distribution_ratios)))
            logging.info("Tri-train to train ratio: {:.5f}".format(self.tritrain_ratio))

        logging.info("Massaging data into input format...")
        self.train_data = self.get_data(train_sentences + tritrain_sentences, True)
        self.dev_data = self.get_data(dev_sentences, False)

        logging.info("Train batches: {}".format(self.train_data[0].shape[0] / self.batch_size))

    def format_distribution(self, distribution):
        return ",".join("{:.5f}".format(p) for p in distribution)

    def get_bucket(self, sentence_length):
        # Don't count <s> and </s>.
        return abs(sentence_length - 1 - 2)/self.bucket_size

    def get_sentence_length_distribution(self, sentences):
        counts = collections.Counter((self.get_bucket(len(s[0])) for s in sentences))
        buckets = [counts[i] for i in range(self.max_tritrain_length/self.bucket_size)]
        buckets_sum = float(sum(buckets))
        return [b/buckets_sum for b in buckets]

    def get_embedding_indexes(self, token):
        return [space.index(space.extract(token)) for space in self.embedding_spaces.values()]

    def get_data(self, sentences, is_training):
        data_size = len(sentences)
        max_tokens = self.train_max_tokens if is_training else self.dev_max_tokens

        data_tokens = np.empty([data_size, max_tokens], dtype=object)
        data_x = np.zeros([data_size, max_tokens, len(self.embedding_spaces)], dtype=np.int32)
        data_y = np.zeros([data_size, max_tokens], dtype=np.int32)
        data_num_tokens = np.zeros([data_size], dtype=np.int64)
        data_tritrain = np.zeros([data_size], dtype=np.float32)
        data_weights = np.zeros([data_size, max_tokens], dtype=np.float32)

        for i,(tokens,supertags,is_tritrain) in enumerate(sentences):
            if len(tokens) != len(supertags):
                raise ValueError("Number of tokens ({}) should match number of supertags ({}).".format(len(tokens), len(supertags)))
            if len(tokens) > max_tokens:
                logging.info("Skipping sentence of length {}.".format(len(tokens)))
                continue

            data_tokens[i,:len(tokens)] = tokens

            x = [self.get_embedding_indexes(t) for t in tokens]
            y = [self.supertag_space.index(s) for s in supertags]

            data_x[i,:len(x)] = x
            data_y[i,:len(y)] = np.absolute(y)

            data_num_tokens[i] = len(x)
            data_tritrain[i] = int(is_tritrain)

            # Labels with negative indices should have 0 weight.
            data_weights[i,:len(y)] = [int(y_val >= 0) for y_val in y]
            if is_training and is_tritrain:
                # Tri-training data is weighted so that the sentence length distribution matches the training data.
                data_weights[i,:len(y)] *= self.distribution_ratios[self.get_bucket(len(x))] * self.tritrain_ratio

        return (data_tokens, data_x, data_y, data_num_tokens, data_tritrain, data_weights)
