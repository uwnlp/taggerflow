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
    parser.add_argument("-r", "--run_name", help="named used to identify logs", default="default")
    parser.add_argument("-g", "--gpu", help="specify gpu devices to use")
    parser.add_argument("-l", "--logdir", help="directory to contain logs", default="logs")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with LoggingToFile(args.logdir, "init.log"):
        data = SupertaggerData()

    configs = expand_grid(args.grid)
    if len(configs) != 1:
        raise ValueError("Only one grid tested for now.")

    for config in configs:
        with LoggingToFile(args.logdir, args.run_name + ".log"):
            model = SupertaggerModel(config, data)
            trainer = SupertaggerTrainer(args.logdir, args.run_name)
            trainer.train(model, data)
