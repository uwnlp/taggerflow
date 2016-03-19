#!/usr/bin/env python
import os
import logging
import threading

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

            population_thread = threading.Thread(target=data.populate_train_queue, args=(session, train_model))
            population_thread.start()

            evaluator = SupertaggerEvaluator(session, data.dev_data, dev_model, train_model.global_step, self.writer, self.logdir)

            i = 0
            epoch = 0
            train_loss = 0.0

            # Evaluator tells us if we should stop.
            while evaluator.maybe_evaluate():
                i += 1

                _, loss = session.run([train_model.optimize,
                                       train_model.loss])
                train_loss += loss
                if i % 100 == 0:
                    timer.tick("{} training steps".format(i))


                if i >= (len(data.train_sentences) + len(data.tritrain_sentences))/data.batch_size:
                    train_loss = train_loss / i
                    logging.info("Epoch {} complete(steps={}, loss={:.3f}).".format(epoch, i, train_loss))
                    self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Train Loss", simple_value=train_loss)]),
                                            tf.train.global_step(session, train_model.global_step))
                    i = 0
                    epoch += 1
                    train_loss = 0.0
