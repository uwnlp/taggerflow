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

def output_supertagger(session, data, model, supertag_space, logdir, pstagged_file):
    tokens,x,y,num_tokens,is_tritrain,weights = data
    with Timer("Dev evaluation"):
        probabilities = session.run(model.probabilities, {
            model.x: x,
            model.num_tokens: num_tokens
        })
    num_correct = np.sum(np.equal(np.argmax(probabilities,2), y)[y >= 0])
    num_total = np.sum(weights)
    accuracy = (100.0 * num_correct)/num_total
    logging.info("Accuracy: {:.3f}% ({}/{})".format(accuracy, num_correct, num_total))

    with open(os.path.join(logdir, pstagged_file), "w") as f:
        for i,n in enumerate(num_tokens):
            for t,p in zip(tokens[1:n-1,i], probabilities[1:n-1,i,:]):
                max_p = max(p)
                unpruned = np.nonzero(np.divide(p,max_p) > 1e-6)[0]
                f.write("{}|{}\n".format(t, "|".join("{}={:.3f}".format(j,math.log(p[j])) for j in unpruned)))
            f.write("\n")

def evaluate_supertagger(session, data, model):
    tokens,x,y,num_tokens,is_tritrain,weights = data
    with Timer("Dev evaluation"):
        prediction = session.run(model.prediction, {
            model.x: x,
            model.num_tokens: num_tokens
        })
    num_correct = np.sum(np.equal(prediction, y)[y >= 0])
    num_total = np.sum(weights)
    accuracy = (100.0 * num_correct)/num_total
    logging.info("Dev accuracy: {:.3f}% ({}/{})".format(accuracy, num_correct, num_total))
    return accuracy

class SupertaggerEvaluationContext(ThreadedContext):
    def __init__(self, session, data, model, global_step, writer, logdir):
        super(SupertaggerEvaluationContext, self).__init__()
        self.session = session
        self.data = data
        self.model = model
        self.global_step = global_step
        self.writer = writer
        self.logdir = logdir
        self.saver = tf.train.Saver(tf.trainable_variables())
        self.best_accuracy = 0.0
        self.evals_without_improvement = 0

    def loop(self):
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
                self.stop = True
                logging.info("Dev accuracy has not improved from {:.3f}% after {} minutes. Stopping training.".format(self.best_accuracy, GRACE_PERIOD))
            else:
                logging.info("{} more minutes without improvement over {:.3f}% permitted.".format(GRACE_PERIOD - self.evals_without_improvement * EVAL_FREQUENCY, self.best_accuracy))
        logging.info("----------------------------")

        summary_values = [tf.Summary.Value(tag="Dev Accuracy", simple_value=accuracy),
                          tf.Summary.Value(tag="Max Dev Accuracy", simple_value=self.best_accuracy)]
        self.writer.add_summary(tf.Summary(value=summary_values), global_step)

        time.sleep(EVAL_FREQUENCY * 60)
