#!/usr/bin/env python

import math
import collections
import logging

import numpy as np

from features import *
from ccgbank import *
from util import *

class SupertaggerData(object):
    min_supertag_count = 10
    min_affix_count = 3
    max_tokens = 100
    batch_size = 512

    def __init__(self):
        train_sentences, dev_sentences = SupertagReader().get_splits()
        logging.info("Train sentences: {}".format(len(train_sentences)))
        logging.info("Dev sentences: {}".format(len(dev_sentences)))

        self.supertag_space = SupertagSpace(train_sentences, min_count=self.min_supertag_count)
        self.embedding_spaces = collections.OrderedDict(
            [("words",    WordSpace(maybe_download("data",
                                                   "http://appositive.cs.washington.edu/resources/",
                                                   "embeddings.raw"))),
             ("prefix_1", PrefixSpace(train_sentences, 1, min_count=self.min_affix_count)),
             ("prefix_2", PrefixSpace(train_sentences, 2, min_count=self.min_affix_count)),
             ("prefix_3", PrefixSpace(train_sentences, 3, min_count=self.min_affix_count)),
             ("prefix_4", PrefixSpace(train_sentences, 4, min_count=self.min_affix_count)),
             ("suffix_1", SuffixSpace(train_sentences, 1, min_count=self.min_affix_count)),
             ("suffix_2", SuffixSpace(train_sentences, 2, min_count=self.min_affix_count)),
             ("suffix_3", SuffixSpace(train_sentences, 3, min_count=self.min_affix_count)),
             ("suffix_4", SuffixSpace(train_sentences, 4, min_count=self.min_affix_count))])

        logging.info("Number of supertags: {}".format(self.supertag_space.size()))
        for name, space in self.embedding_spaces.items():
            logging.info("Number of {}: {}".format(name, space.size()))

        logging.info("Massaging data into training format...")

        self.train_data = self.get_data(train_sentences)
        self.dev_data = self.get_data(dev_sentences)

        logging.info("Train batches: {}".format(self.train_data[0].shape[0] / self.batch_size))
        logging.info("Dev batches: {}".format(self.dev_data[0].shape[0] / self.batch_size))

    def get_embedding_indexes(self, token):
        return [space.index(space.extract_from_token(token)) for space in self.embedding_spaces.values()]

    def get_train_batches(self):
        return self.get_batches(self.train_data)

    def get_dev_batches(self):
        return self.get_batches(self.dev_data)

    def get_batches(self, data):
        data_x, data_y, data_num_tokens, data_mask = data
        batch_size = self.batch_size
        data_size = data_x.shape[0]
        if data_size % batch_size != 0:
            raise ValueError("The data size should be divisible by the batch size.")

        indexes = np.arange(data_size)
        np.random.shuffle(indexes)
        batches = []
        for i in range(data_size / batch_size):
            batch_indexes = indexes[i * batch_size: (i + 1) * batch_size]
            batches.append((data_x[batch_indexes,:,:],
                            data_y[batch_indexes,:],
                            data_num_tokens[batch_indexes],
                            data_mask[batch_indexes,:]))
        return batches

    def get_data(self, sentences):
        sentences = [([self.get_embedding_indexes(t) for t in tokens], [self.supertag_space.index(s) for s in supertags]) for tokens, supertags in sentences]

        # Make the data size divisible by the batch size.
        data_size = int(self.batch_size * math.ceil(len(sentences)/float(self.batch_size)))
        data_x = np.zeros([data_size, self.max_tokens, len(self.embedding_spaces)], dtype=np.int32)
        data_y = np.zeros([data_size, self.max_tokens], dtype=np.int32)
        data_num_tokens = np.zeros([data_size], dtype=np.int64)
        data_mask = np.zeros([data_size, self.max_tokens], dtype=np.float32)

        for i,(x,y) in enumerate(sentences):
            if len(x) != len(y):
                raise ValueError("Number of tokens should match number of supertags.")
            if len(x) > self.max_tokens:
                logging.info("Skipping sentence of length {}.".format(len(x)))
                continue
            data_x[i,:len(x):] = x
            data_y[i,:len(y)] = y
            data_num_tokens[i] = len(x)
            data_mask[i,:len(y)] = y >= 0

        return (data_x, data_y, data_num_tokens, data_mask)
