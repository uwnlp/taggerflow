#!/usr/bin/env python

import time
import logging

import tensorflow as tf

import util

class SupertaggerEvaluationContext(util.ThreadedContext):
    def __init__(self, session, data, model, global_step, writer):
        super(SupertaggerEvaluationContext, self).__init__()
        self.session = session
        self.data = data
        self.model = model
        self.global_step = global_step
        self.writer = writer

    def loop(self):
        time.sleep(120)
        with util.Timer("Dev evaluation"):
            total_correct = 0
            total_total = 0
            for x,y,num_tokens,mask in self.data:
                num_correct, num_total = self.session.run([self.model.num_correct, self.model.num_total], {
                    self.model.x: x,
                    self.model.y: y,
                    self.model.num_tokens: num_tokens,
                    self.model.keep_probability: 1.0
                })
                total_correct += num_correct
                total_total += num_total
            accuracy = (100.0 * total_correct) / total_total
            self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Dev Accuracy", simple_value=accuracy)]),
                                    tf.train.global_step(self.session, self.global_step))
            logging.info("Dev accuracy: {:.3f}% ({}/{})".format(accuracy, total_correct, total_total))
