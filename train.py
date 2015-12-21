#!/usr/bin/env python

import os
import logging

import tensorflow as tf

from evaluation import *
from util import *
from model import *
from config import *

class SupertaggerTrainer(object):

    def __init__(self, logdir):
        self.writer = tf.train.SummaryWriter(logdir, flush_secs=20)

    def train(self, model, data):
        dev_batches = data.get_dev_batches()

        with tf.Session() as session, tf.variable_scope("model", initializer=model.initializer), Timer("Training") as timer:
            tf.initialize_all_variables().run()
            session.run(model.initialize)

            with SupertaggerEvaluationContext(session, dev_batches, model, self.writer) as eval_context:
                epoch = 0
                train_batches = data.get_train_batches()
                while not eval_context.stop:
                    logging.info("========= Epoch {:02d} =========".format(epoch))
                    train_cost = 0.0
                    train_reg = 0.0
                    for i,(x,y,num_tokens,mask) in enumerate(train_batches):
                        if eval_context.stop:
                            break
                        _, cost, reg = session.run([model.optimize, model.cost, model.regularization], {
                            model.x: x,
                            model.y: y,
                            model.num_tokens: num_tokens,
                            model.mask: mask,
                            model.input_dropout_probability: model.config.input_dropout_probability,
                            model.dropout_probability: model.config.dropout_probability
                        })
                        train_cost += cost
                        train_reg += reg
                        if i % 10 == 0:
                            logging.info("{}/{} training steps taken.".format(i+1,len(train_batches)))

                    train_cost = train_cost / len(train_batches)
                    train_reg = train_reg / len(train_batches)
                    self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Train Cost", simple_value=train_cost)]),
                                            tf.train.global_step(session, model.global_step))
                    self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Regularization", simple_value=train_reg)]),
                                            tf.train.global_step(session, model.global_step))
                    logging.info("Epoch mean training cost: {:.3f}".format(train_cost))
                    logging.info("Epoch mean training regularization: {:.3f}".format(train_reg))
                    timer.tick("Epoch {}".format(epoch))
                    epoch += 1
                    logging.info("============================")
