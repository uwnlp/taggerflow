import time
import logging
import os
import math
import numpy as np
import tensorflow as tf

from util import *
from model import *

# Evaluate every 2 minutes.
EVAL_FREQUENCY = 2

# Allow the model 30 chances and about 60 minutes to improve.
#GRACE_PERIOD = 60

# Run basically forever.
GRACE_PERIOD = 10000

def evaluate_supertagger(session, data, model):
    x,y,num_tokens,is_tritrain,weights = data
    with Timer("Dev evaluation"):
        scores = session.run(model.scores, {
            model.x: x,
            model.num_tokens: num_tokens
        })
    predictions = np.argmax(scores, 2)
    num_correct = np.sum(np.equal(predictions, y) * weights)
    num_total = np.sum(weights)
    accuracy = (100.0 * num_correct)/num_total
    logging.info("Accuracy: {:.3f}% ({}/{})".format(accuracy, num_correct, num_total))
    return accuracy

class SupertaggerEvaluator(object):
    def __init__(self, session, data, model, global_step, writer, logdir):
        self.session = session
        self.data = data
        self.model = model
        self.global_step = global_step
        self.writer = writer
        self.logdir = logdir
        self.saver = tf.train.Saver(tf.trainable_variables())
        self.best_accuracy = 0.0
        self.evals_without_improvement = 0
        self.last_eval = time.time()

    def maybe_evaluate(self):
        if time.time() - self.last_eval > EVAL_FREQUENCY * 60:
            global_step = tf.train.global_step(self.session, self.global_step)
            logging.info("----------------------------")
            logging.info("Evaluating at step {}.".format(global_step))
            accuracy = evaluate_supertagger(self.session, self.data, self.model)

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.evals_without_improvement = 0
                logging.info("New max dev accuracy: {:.3f}%".format(self.best_accuracy))
                with Timer("Saving model"):
                    save_path = self.saver.save(self.session, os.path.join(self.logdir, "model.ckpt"), global_step)
                    logging.info("Model saved in file: %s" % save_path)
            else:
                self.evals_without_improvement += 1
                if self.evals_without_improvement * EVAL_FREQUENCY >= GRACE_PERIOD:
                    logging.info("Dev accuracy has not improved from {:.3f}% after {} minutes. Stopping training.".format(self.best_accuracy, GRACE_PERIOD))
                    return False
                else:
                    logging.info("{} more minutes without improvement over {:.3f}% permitted.".format(GRACE_PERIOD - self.evals_without_improvement * EVAL_FREQUENCY, self.best_accuracy))
            logging.info("----------------------------")

            summary_values = [tf.Summary.Value(tag="Dev Accuracy", simple_value=accuracy),
                              tf.Summary.Value(tag="Max Dev Accuracy", simple_value=self.best_accuracy)]
            self.writer.add_summary(tf.Summary(value=summary_values), global_step)
            self.last_eval = time.time()
        return True
