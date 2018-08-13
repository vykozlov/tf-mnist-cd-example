# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import os
import sys
import math

from collections import namedtuple
from datetime import datetime

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import data.storeincsv as incsv
import models.deepcnn as cnn


FLAGS = None

ParamHeader = ['Timestamp', 'Script', 'Info', 'Batch_size', 'Num_steps',
               'TestAccuracy', 'TotalTime', 'TestTime', 'TrainTime',
               'MeanPerBatch', 'StDev']
ParamEntry = namedtuple('ParamEntry', ParamHeader)

def main(_):
    """ Main function to initialize and run neural network """
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv, keep_prob = cnn.deepcnn(x)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    param_entries = []
    param_entries.append(ParamHeader)
    check_step = 1000
    mnist_batchsize = 50
    mnist_steps = 20000
    num_steps_burn_in = 10

    if FLAGS.with_profiling:
        mnist_steps = 2
        num_steps_burn_in = 0
        print("=> Profiling is enabled!")

    if FLAGS.mnist_batch > -1:
        mnist_batchsize = FLAGS.mnist_batch

    if FLAGS.mnist_steps > -1:
        mnist_steps = FLAGS.mnist_steps

    print("mnist_steps: ", mnist_steps)

    print("Ready for training, start time counting")
    # start time
    start = time.time()

    train_duration = 0.0
    train_duration_squared = 0.0
    check_duration = 0.0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch0 = mnist.train.next_batch(mnist_batchsize) # for burn_in we can use a fixed batch
        for i in range(mnist_steps + num_steps_burn_in):
            if i == num_steps_burn_in:
                burn_in_end = time.time()
                tcheck_prev = burn_in_end

            batch = mnist.train.next_batch(mnist_batchsize) if (i >= num_steps_burn_in) else batch0

            if (i >= num_steps_burn_in) and (i - num_steps_burn_in) % check_step == 0:
                tcheck = time.time()
                train_accuracy = accuracy.eval(feed_dict={x: batch[0],
                                                          y_: batch[1],
                                                          keep_prob: 1.0})
                dtcheck = tcheck - tcheck_prev
                nbatches = check_step if (i - num_steps_burn_in) > 0 else 0
                t1batch = dtcheck/float(nbatches) if nbatches > 0 else 0
                print("step {0:6d}, training accuracy {1:5.3f}"\
                      "({2:5d} batches trained in {3:6.4f} s, i.e. {4:9.07f} s/batch)"
                      .format(i - num_steps_burn_in, train_accuracy, nbatches, dtcheck, t1batch))
                tcheck_prev = time.time()
                check_duration += (time.time() - tcheck)

            start_train = time.time()              # measure training time per batch

            if FLAGS.with_profiling:
                run_metadata = tf.RunMetadata()
                train_step_ = sess.run(train_step, feed_dict={x: batch[0],
                                                              y_: batch[1],
                                                              keep_prob: 0.5},
                                       options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                       run_metadata=run_metadata)
            else:
                train_step_ = sess.run(train_step, feed_dict={x: batch[0],
                                                              y_: batch[1],
                                                              keep_prob: 0.5})

            duration = time.time() - start_train   # measure training time per batch
            if i >= num_steps_burn_in:
                train_duration += duration
                train_duration_squared += duration * duration

        start_test = time.time()
        param_accuracy = accuracy.eval(feed_dict={x: mnist.test.images,
                                                  y_: mnist.test.labels,
                                                  keep_prob: 1.0})
        test_duration = time.time() - start_test
        total_runtime = time.time() - burn_in_end

        mn = train_duration / mnist_steps
        vr = train_duration_squared / mnist_steps - mn * mn
        sd = math.sqrt(vr)
        print('test accuracy %g' % param_accuracy)
        print('run in %g s, test: %g s, checks: %g s, train: %g s, burn_in: %g s' %
              (total_runtime, test_duration, check_duration, train_duration, burn_in_end - start))
        print('mean per batch %g +/- %g s' % (mn, sd))
        param_entries.append(ParamEntry(
            datetime.now(), os.path.basename(__file__),
            "", mnist_batchsize, mnist_steps, param_accuracy,
            total_runtime, test_duration, train_duration, mn, sd))

        # Dump profiling data (*)
        if FLAGS.with_profiling:
            ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
            opts = ProfileOptionBuilder(
                ProfileOptionBuilder.time_and_memory()).with_node_names().build()
            tf.profiler.profile(tf.get_default_graph(),
                                run_meta=run_metadata,
                                cmd='code',
                                options=opts)

#      prof_timeline = tf.python.client.timeline.Timeline(run_metadata.step_stats)
#      prof_ctf = prof_timeline.generate_chrome_trace_format()
#      with open('./prof_ctf.json', 'w') as fp:
#          print("Dumped to prof_ctf.json")
#          fp.write(prof_ctf)

    if FLAGS.csv_file:
        incsv.store_data_in_csv(FLAGS.csv_file, param_entries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument("--mnist_batch", type=int, default=-1,
                        help="Batch size")
    parser.add_argument("--mnist_steps", type=int, default=-1,
                        help="Number of steps to train")
    parser.add_argument("--with_profiling", nargs='?', const=True, type=bool, default=False,
                        help="(experimental) Enable profiling. "\
                             "If --mnist_steps is not specified, only 2 epochs are processed!")
    parser.add_argument('--csv_file', type=str,
                        default='',
                        help="File (.csv) to output script results. "\
                             "If no file is passed in, csv file will not be created.")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
