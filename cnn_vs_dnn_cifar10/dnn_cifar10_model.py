"""Convolutional Neural Network Estimator for cifar-10, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def dnn_model_fn(features, labels, mode):
  """Model function for DNN."""
  # Input Layer
  # cifar-10 images are 32x32 pixels, and have 3 color channel
  input_layer = tf.reshape(features["x"], [-1, 32 * 32 * 3])


  # hidden layer 1
  # 1024 nodes
  hidden1 = tf.layers.dense(inputs=input_layer, units=1024)

  # hidden layer 2
  # 1024 nodes
  hidden2 = tf.layers.dense(inputs=hidden1, units=1024)

  # hidden layer 3
  # 1024 nodes
  hidden3 = tf.layers.dense(inputs=hidden2, units=1024)

  # dropout layer for training, prevents overfitting
  dropout = tf.layers.dropout(inputs=hidden3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
