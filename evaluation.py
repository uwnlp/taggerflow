#!/usr/bin/env python

import time
import logging

import numpy as np
import tensorflow as tf

from util import *

# Evaluate every 2 minutes.
EVAL_FREQUENCY = 2

# Allow the model 20 chances and about 40 minutes to improve.
#GRACE_PERIOD = 40

# Run basically forever.
GRACE_PERIOD = 10000

class SupertaggerEvaluationContext(ThreadedContext):
    def __init__(self, session, data, model, writer):
        super(SupertaggerEvaluationContext, self).__init__()
        self.session = session
        self.data = data
        self.model = model
        self.writer = writer
        self.best_accuracy = 0.0
        self.evals_without_improvement = 0

    def loop(self):
        x,y,num_tokens,mask = self.data
        with Timer("Dev evaluation"):
            prediction = self.session.run(self.model.prediction, {
                self.model.x: x,
                self.model.num_tokens: num_tokens
            })
        num_correct = 0
        for i,n in enumerate(num_tokens):
            num_correct += sum(int(prediction[i,j] == y[i,j]) for j in range(n) if y[i,j] >= 0)
        num_total = np.sum(mask)
        accuracy = (100.0 * num_correct)/num_total
        global_step = tf.train.global_step(self.session, self.model.global_step)

        logging.info("----------------------------")
        logging.info("Evaluating at step {}.".format(global_step))
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

        summary_values = [tf.Summary.Value(tag="Dev Accuracy", simple_value=accuracy),
                          tf.Summary.Value(tag="Max Dev Accuracy", simple_value=self.best_accuracy)]
        self.writer.add_summary(tf.Summary(value=summary_values), global_step)

        time.sleep(EVAL_FREQUENCY * 60)
