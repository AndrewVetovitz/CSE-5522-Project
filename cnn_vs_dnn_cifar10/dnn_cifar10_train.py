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
  # Load training data
  train_input_fn = cifar10_loader.input_func(False, os.path.join(DATA_PARENT_DIR, DATA_DIR), 100)

  # Create the Estimator
  cifar10_classifier = tf.estimator.Estimator(
      model_fn=dnn_model_fn, model_dir="dnn_cifar10_model_dir")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  cifar10_classifier.train(
    input_fn=train_input_fn,
    steps=10000,
    hooks=[logging_hook])


if __name__ == "__main__":
  tf.app.run()
