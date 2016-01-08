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
        self.logdir = logdir
        self.writer = tf.train.SummaryWriter(logdir, flush_secs=20)

    def train(self, config, data, params):
        with tf.Session() as session, Timer("Training") as timer:
            with tf.variable_scope("model", initializer=custom_init_ops.dyer_initializer()):
                train_model = SupertaggerModel(config, data, is_training=True)

            with tf.variable_scope("model", reuse=True):
                dev_model = SupertaggerModel(config, data, is_training=False)

            session.run(tf.initialize_all_variables())

            with tf.variable_scope("model", reuse=True):
                params.assign_pretrained(session)

            with SupertaggerEvaluationContext(session, data.dev_data, dev_model, train_model.global_step, self.writer, self.logdir) as eval_context:
                i = 0
                epoch = 0
                train_cost = 0.0
                train_reg = 0.0
                session.run(train_model.input_queue_refresh)
                while not eval_context.stop:
                    i += 1
                    _, cost, reg, input_queue_size = session.run([train_model.optimize,
                                                                  train_model.cost,
                                                                  train_model.regularization,
                                                                  train_model.input_queue_size])
                    train_cost += cost
                    train_reg += reg
                    if i % 10 == 0:
                        timer.tick("{} training steps".format(i))
                        logging.info("Remaining sentences: {}".format(input_queue_size))
                    if input_queue_size < data.batch_size * 2:
                        train_cost = train_cost / i
                        train_reg = train_reg / i
                        logging.info("Epoch {} complete(steps={}, cost={:.3f}, regularization={:.3f}).".format(epoch, i, train_cost, train_reg))
                        self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Train Cost", simple_value=train_cost)]),
                                                tf.train.global_step(session, train_model.global_step))
                        self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Regularization", simple_value=train_reg)]),
                                                tf.train.global_step(session, train_model.global_step))
                        epoch += 1
                        train_cost = 0.0
                        train_reg = 0.0
                        session.run(train_model.input_queue_refresh)
