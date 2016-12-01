#!/usr/bin/env python

import sys
import os
import argparse
import logging
import tempfile
from train import *
from model import *
from data import *
from config import *
from util import *
from parameters import *

from tensorflow.python.framework import graph_util

def get_pretrained_parameters(params_file):
    params = Parameters()
    params.read(params_file)
    return params

def get_default_parameters(sentences):
    parameters = Parameters([("words",    TurianEmbeddingSpace(maybe_download("data",
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
    return parameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("grid", help="grid json file")
    parser.add_argument("-e", "--exp", help="named used to the identify set of experiments", default="default")
    parser.add_argument("-g", "--gpu", help="specify gpu devices to use")
    parser.add_argument("-l", "--logdir", help="directory to contain logs", default="logs")
    parser.add_argument("-p", "--params", help="pretrained parameter file")
    parser.add_argument("-t", "--tritrain", help="whether or not to use tritraining data", action="store_true")
    parser.add_argument("-c", "--checkpoint", help="recover checkpoint, evaluate, and output frozen graph")
    parser.add_argument("-j", "--jackknifed", help="all siblings are used as training data instead")

    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    stream_handler = logging.StreamHandler()
    logging.getLogger().addHandler(stream_handler)
    logging.getLogger().setLevel(logging.INFO)

    exp_logdir = os.path.join(args.logdir, args.exp)

    maybe_mkdirs(exp_logdir)

    output_dir = tempfile.mkdtemp(prefix="taggerflow-")

    with LoggingToFile(exp_logdir, "init.log"):
        supertag_space = SupertagSpace(maybe_download("data",
                                                      "http://appositive.cs.washington.edu/resources/",
                                                      "categories"))

        reader = SupertagReader()
        train_sentences, tritrain_sentences, dev_sentences = reader.get_splits(args.tritrain and args.checkpoint is None)

        if args.params is None:
            parameters = get_default_parameters(lambda:itertools.chain(train_sentences, tritrain_sentences))
        else:
            parameters = get_pretrained_parameters(args.params)
        parameters.write(output_dir)
        if args.jackknifed is not None:
            logging.info("Replacing training data with siblings of {}".format(args.jackknifed))
            jackknifed_dir = os.path.dirname(args.jackknifed)
            train_sentences = []
            for filename in os.listdir(jackknifed_dir):
                filepath = os.path.join(jackknifed_dir, filename)
                if filename.endswith(".stagged") and filepath != args.jackknifed:
                    logging.info("Adding {}".format(filepath))
                    train_sentences.extend(reader.get_sentences(filepath, False))

        data = SupertaggerData(supertag_space, parameters.embedding_spaces, train_sentences, tritrain_sentences, dev_sentences)

    if args.checkpoint is not None:
        logging.info("Restoring from: {}".format(args.checkpoint))

    if args.checkpoint is not None or args.params is not None:
        # Evaluate the full size as a sanity check.
        with tf.Session() as session:
            with tf.variable_scope("model"):
                model = SupertaggerModel(None, data, is_training=False)
            with tf.variable_scope("model", reuse=True):
                parameters.assign_pretrained(session)
            if args.checkpoint is not None:
                saver = tf.train.Saver()
                saver.restore(session, args.checkpoint)
            evaluate_supertagger(session, data.dev_data, model)

        # Write the smaller graph in protobuffer format.
        g = tf.Graph()
        with g.as_default(), tf.Session() as session:
            with g.name_scope("frozen"), tf.variable_scope("model"):
                model = SupertaggerModel(None, data, is_training=False, max_tokens=72)
            with g.name_scope("frozen"), tf.variable_scope("model", reuse=True):
                parameters.assign_pretrained(session)
            if args.checkpoint is not None:
                saver = tf.train.Saver()
                saver.restore(session, args.checkpoint)
            tf.train.write_graph(graph_util.convert_variables_to_constants(session,
                                                                           g.as_graph_def(),
                                                                           ["frozen/model/prediction/scores"]),
                                 output_dir,
                                 "graph.pb",
                                 as_text=False)
        logging.info("Computation graph written to {}/graph.pb".format(output_dir))

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
