#!/usr/bin/env python

import os
import argparse
import logging

from train import *
from model import *
from data import *
from config import *
from util import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("grid", help="grid json file")
    parser.add_argument("-e", "--exp", help="named used to the identify set of experiments", default="default")
    parser.add_argument("-g", "--gpu", help="specify gpu devices to use")
    parser.add_argument("-l", "--logdir", help="directory to contain logs", default="logs")
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
        data = SupertaggerData()
        configs = expand_grid(args.grid)

    for config in configs:
        run_logdir = os.path.join(exp_logdir, config.name)
        if not os.path.exists(run_logdir):
            os.makedirs(run_logdir)

        with LoggingToFile(run_logdir, "info.log"):
            stream_handler.setFormatter(logging.Formatter("{} - %(message)s".format(config.name)))
            with tf.Graph().as_default():
                model = SupertaggerModel(config, data)
                trainer = SupertaggerTrainer(run_logdir)
                trainer.train(model, data)
