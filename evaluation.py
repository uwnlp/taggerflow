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
            num_correct = 0
            num_total = 0
            for x,y,num_tokens in self.data:
                prediction = self.session.run(self.model.prediction, {
                    self.model.x: x,
                    self.model.num_tokens: num_tokens,
                    self.model.keep_probability: 1.0
                })
                for i,n in enumerate(num_tokens):
                    num_total += n
                    num_correct += sum(int(prediction[i,j] == y[i,j]) for j in range(n))

            accuracy = (100.0 * num_correct)/num_total
            self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Dev Accuracy", simple_value=accuracy)]),
                                    tf.train.global_step(self.session, self.global_step))
            logging.info("Dev accuracy: {:.3f}% ({}/{})".format(accuracy, num_correct, num_total))
