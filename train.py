#!/usr/bin/env python

import os
import logging

import tensorflow as tf
import custom_init_ops

from evaluation import *
from util import *
from model import *
from config import *

class SupertaggerTrainer(object):

    def __init__(self, logdir):
        self.writer = tf.train.SummaryWriter(logdir, flush_secs=20)

    def train(self, config, data, params):
        with tf.Session() as session, Timer("Training") as timer:
            with tf.variable_scope("model", initializer=custom_init_ops.dyer_initializer()):
                train_model = SupertaggerModel(config, data, batch_size=data.batch_size, is_training=True)

            with tf.variable_scope("model", reuse=True):
                dev_model = SupertaggerModel(config, data, batch_size=data.dev_data[0].shape[0], is_training=False)

            session.run(tf.initialize_all_variables())

            with tf.variable_scope("model", reuse=True):
                params.assign_pretrained(session)

            with SupertaggerEvaluationContext(session, data.dev_data, dev_model, self.writer) as eval_context:
                epoch = 0
                train_batches = data.get_batches(data.train_data)
                while not eval_context.stop:
                    logging.info("========= Epoch {:02d} =========".format(epoch))
                    train_cost = 0.0
                    train_reg = 0.0
                    for i,(x,y,num_tokens,is_tritrain,weights) in enumerate(train_batches):
                        if eval_context.stop:
                            break
                        _, cost, reg = session.run([train_model.optimize, train_model.cost, train_model.regularization], {
                            train_model.x: x,
                            train_model.y: y,
                            train_model.num_tokens: num_tokens,
                            train_model.tritrain: is_tritrain,
                            train_model.weights: weights
                        })
                        train_cost += cost
                        train_reg += reg
                        if i % 10 == 0:
                            timer.tick("{}/{} training steps".format(i+1,len(train_batches)))

                    train_cost = train_cost / len(train_batches)
                    train_reg = train_reg / len(train_batches)
                    self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Train Cost", simple_value=train_cost)]),
                                            tf.train.global_step(session, train_model.global_step))
                    self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Regularization", simple_value=train_reg)]),
                                            tf.train.global_step(session, train_model.global_step))
                    logging.info("Epoch mean training cost: {:.3f}".format(train_cost))
                    logging.info("Epoch mean training regularization: {:.3f}".format(train_reg))
                    timer.tick("Epoch {}".format(epoch))
                    epoch += 1
                    logging.info("============================")
