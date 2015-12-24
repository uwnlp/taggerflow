#!/usr/bin/env python

import os
import argparse
import logging

from train import *
from model import *
from data import *
from config import *
from util import *
from parameters import *

def get_pretrained_parameters(params_file):
    params = Parameters()
    params.read(params_file)
    return params

def get_default_parameters(sentences):
    return Parameters([("words",    TurianEmbeddingSpace(maybe_download("data",
                                                                        "http://appositive.cs.washington.edu/resources/",
                                                                        "embeddings.raw"))),
                       ("prefix_1", EmpiricalPrefixSpace(1, sentences)),
                       ("prefix_2", EmpiricalPrefixSpace(2, sentences)),
                       ("prefix_3", EmpiricalPrefixSpace(3, sentences)),
                       ("prefix_4", EmpiricalPrefixSpace(4, sentences)),
                       ("suffix_1", EmpiricalSuffixSpace(1, sentences)),
                       ("suffix_2", EmpiricalSuffixSpace(2, sentences)),
                       ("suffix_3", EmpiricalSuffixSpace(3, sentences)),
                       ("suffix_4", EmpiricalSuffixSpace(4, sentences))])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("grid", help="grid json file")
    parser.add_argument("-e", "--exp", help="named used to the identify set of experiments", default="default")
    parser.add_argument("-g", "--gpu", help="specify gpu devices to use")
    parser.add_argument("-l", "--logdir", help="directory to contain logs", default="logs")
    parser.add_argument("-p", "--params", help="pretrained parameter file")
    parser.add_argument("-t", "--tritrain", help="whether or not to use tri-training data instead", action="store_true")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    stream_handler = logging.StreamHandler()
    logging.getLogger().addHandler(stream_handler)
    logging.getLogger().setLevel(logging.INFO)


    exp_logdir = os.path.join(args.logdir, args.exp)

    if not os.path.exists(exp_logdir):
        os.makedirs(exp_logdir)

    with LoggingToFile(exp_logdir, "init.log"):
        train_sentences, dev_sentences = SupertagReader().get_splits(args.tritrain)
        supertag_space = SupertagSpace(maybe_download("data",
                                                      "http://appositive.cs.washington.edu/resources/",
                                                      "categories"))
        if args.params is None:
            parameters = get_default_parameters(train_sentences)
        else:
            parameters = get_pretrained_parameters(args.params)
        data = SupertaggerData(supertag_space, parameters.embedding_spaces, train_sentences, dev_sentences)
        configs = expand_grid(args.grid)

    for config in configs:
        run_logdir = os.path.join(exp_logdir, config.name)
        if not os.path.exists(run_logdir):
            os.makedirs(run_logdir)

        with LoggingToFile(run_logdir, "info.log"):
            stream_handler.setFormatter(logging.Formatter("{} - %(message)s".format(config.name)))
            with tf.Graph().as_default():
                trainer = SupertaggerTrainer(run_logdir)
                trainer.train(config, data, parameters)
