"""Convolutional Neural Network Estimator for cifar-10, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import os

from dnn_cifar10_model import dnn_model_fn
import cifar10_loader

DATA_PARENT_DIR = "cifar10-data"
DATA_DIR = "cifar-10-batches-bin"

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
  # Load training and eval data
  eval_input_fn = cifar10_loader.input_func(True, os.path.join(DATA_PARENT_DIR, DATA_DIR), 100)

  # Create the Estimator
  cifar10_classifier = tf.estimator.Estimator(
      model_fn=dnn_model_fn, model_dir="dnn_cifar10_model_dir")


  while True:
      eval_results = cifar10_classifier.evaluate(input_fn=eval_input_fn, steps=3)
      print(eval_results)
      time.sleep(3*60) #eval once every 3 minutes, which is how often model checkpoints are saved during training


if __name__ == "__main__":
  tf.app.run()
