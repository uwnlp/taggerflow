#!/usr/bin/env python

import math
import collections
import logging
import itertools

import numpy as np
import random

from features import *
from ccgbank import *
from util import *

class SupertaggerData(object):
    max_tokens = 102
    batch_size = 32
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
            self.tritrain_ratio = 15
        else:
            self.tritrain_ratio = 1

        logging.info("Massaging data into input format...")
        self.train_sentences = train_sentences
        self.tritrain_sentences = tritrain_sentences
        self.dev_data = self.get_data(dev_sentences)

    def format_distribution(self, distribution):
        return ",".join("{:.5f}".format(p) for p in distribution)

    def get_bucket(self, sentence_length):
        # Don't count <s> and </s>.
        return max(sentence_length - 1 - 2, 0)/self.bucket_size

    def get_sentence_length_distribution(self, sentences):
        counts = collections.Counter((self.get_bucket(len(s[0])) for s in sentences))
        buckets = [counts[i] for i in range(self.max_tritrain_length/self.bucket_size)]
        buckets_sum = float(sum(buckets))
        return [b/buckets_sum for b in buckets]

    def get_embedding_indexes(self, token):
        return [space.index(space.extract(token)) for space in self.embedding_spaces.values()]

    def tensorize(self, sentence):
        tokens, supertags, is_tritrain = sentence

        if len(tokens) != len(supertags):
            raise ValueError("Number of tokens ({}) should match number of supertags ({}).".format(len(tokens), len(supertags)))
        if len(tokens) > self.max_tokens:
            logging.info("Skipping sentence of length {}.".format(len(tokens)))
            return None

        x = np.array([self.get_embedding_indexes(t) for t in tokens])
        y = np.array([self.supertag_space.index(s) for s in supertags])

        # Labels with negative indices should have 0 weight.
        weights = (y >= 0).astype(float)

        x.resize([self.max_tokens, x.shape[1]])
        y.resize([self.max_tokens])
        weights.resize([self.max_tokens])
        return x, np.absolute(y), len(tokens), int(is_tritrain), weights

    def populate_train_queue(self, session, model):
        i = 0
        tritrain_probability = len(self.tritrain_sentences)/float(len(self.tritrain_sentences) + 15)
        while True:
            if np.random.rand() > tritrain_probability:
                s = random.choice(self.train_sentences)
            else:
                s = random.choice(self.tritrain_sentences)
            tensors = self.tensorize(s)
            if tensors is not None:
                session.run(model.input_enqueue, { i:t for i,t in zip(model.inputs, tensors) })
                i += 1
                if i % 10000 == 0:
                    logging.info("Queued {} sentences.".format(i))

    def get_data(self, sentences):
        tensors = (self.tensorize(s) for s in sentences)
        results = [np.array(v) for v in zip(*(t for t in tensors if t is not None))]
        return results
