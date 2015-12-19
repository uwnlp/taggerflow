#!/usr/bin/env python

import time
import logging

import numpy as np
import tensorflow as tf

import util

# Evaluate every 2 minutes.
EVAL_FREQUENCY = 2

# Allow the model 30 minutes to improve.
GRACE_PERIOD = 30

class SupertaggerEvaluationContext(util.ThreadedContext):
    def __init__(self, session, data, model, global_step, writer):
        super(SupertaggerEvaluationContext, self).__init__()
        self.session = session
        self.data = data
        self.model = model
        self.global_step = global_step
        self.writer = writer
        self.best_accuracy = 0.0
        self.evals_without_improvement = 0

    def loop(self):
        time.sleep(EVAL_FREQUENCY * 60)
        with util.Timer("Dev evaluation"):
            num_correct = 0
            num_total = 0
            for x,y,num_tokens,mask in self.data:
                prediction = self.session.run(self.model.prediction, {
                    self.model.x: x,
                    self.model.num_tokens: num_tokens,
                    self.model.keep_probability: 1.0
                })
                for i,n in enumerate(num_tokens):
                    num_correct += sum(int(prediction[i,j] == y[i,j]) for j in range(n) if y[i,j] >= 0)
                num_total += np.sum(mask)
            accuracy = (100.0 * num_correct)/num_total

        self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Dev Accuracy", simple_value=accuracy)]),
                                tf.train.global_step(self.session, self.global_step))
        self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Max Dev Accuracy", simple_value=self.best_accuracy)]),
                                tf.train.global_step(self.session, self.global_step))

        logging.info("----------------------------")
        logging.info("Dev accuracy: {:.3f}% ({}/{})".format(accuracy, num_correct, num_total))

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.evals_without_improvement = 0
            logging.info("New max dev accuracy: {:.3f}%".format(self.best_accuracy))
        else:
            self.evals_without_improvement += 1
            if self.evals_without_improvement * EVAL_FREQUENCY >= GRACE_PERIOD:
                self.stop = True
                logging.info("Dev accuracy has not improved from {:.3f}% after {} minutes. Stopping training.".format(self.best_accuracy, GRACE_PERIOD))
            else:
                logging.info("{} more minutes without improvement over {:.3f}% permitted.".format(GRACE_PERIOD - self.evals_without_improvement * EVAL_FREQUENCY, self.best_accuracy))
        logging.info("----------------------------")
