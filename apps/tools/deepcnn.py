# -*- coding: utf-8 -*-
"""
Simple convolutional neural network

Created on Mon Jul  2 15:43:01 2018

@author: valentin
"""

import tensorflow as tf
import tools.nnet_funcs as nnfunc

def deepcnn(x):
    """deepcnn builds the graph for a deep net for classifying digits.

    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.

    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = nnfunc.weight_variable([5, 5, 1, 32])
    b_conv1 = nnfunc.bias_variable([32])
    h_conv1 = tf.nn.relu(nnfunc.conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = nnfunc.max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = nnfunc.weight_variable([5, 5, 32, 64])
    b_conv2 = nnfunc.bias_variable([64])
    h_conv2 = tf.nn.relu(nnfunc.conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = nnfunc.max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = nnfunc.weight_variable([7 * 7 * 64, 1024])
    b_fc1 = nnfunc.bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = nnfunc.weight_variable([1024, 10])
    b_fc2 = nnfunc.bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob
    