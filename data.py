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

        logging.info("Massaging data into mini-batch format...")

        self.train_batches = self.get_batches(train_sentences)
        self.dev_batches = self.get_batches(dev_sentences)

        logging.info("Train batches: {}".format(len(self.train_batches)))
        logging.info("Dev batches: {}".format(len(self.dev_batches)))

    def get_embedding_indexes(self, token):
        return [space.index(space.extract_from_token(token)) for space in self.embedding_spaces.values()]

    def get_batches(self, sentences):
        data = [([self.get_embedding_indexes(t) for t in tokens], [self.supertag_space.index(s) for s in supertags]) for tokens, supertags in sentences]

        batch_size = self.batch_size
        batches = []
        num_batches = int(math.ceil(len(data)/float(batch_size)))
        for i in range(num_batches):
            batch_x = np.zeros([batch_size, self.max_tokens, len(self.embedding_spaces)], dtype=np.int32)
            batch_y = np.zeros([batch_size, self.max_tokens], dtype=np.int32)
            batch_num_tokens = np.zeros([batch_size], dtype=np.int64)
            batch_mask = np.zeros([batch_size, self.max_tokens], dtype=np.float32)
            for j,(x,y) in enumerate(data[i * batch_size: (i + 1) * batch_size]):
                if len(x) > self.max_tokens:
                    logging.info("Skipping sentence of length {}.".format(len(x)))
                    continue
                batch_x[j,:len(x):] = x
                batch_y[j,:len(y)] = y
                batch_num_tokens[j] = len(x)
                batch_mask[j,:len(y)] = y >= 0
            batches.append((batch_x, batch_y, batch_num_tokens, batch_mask))
        return batches
