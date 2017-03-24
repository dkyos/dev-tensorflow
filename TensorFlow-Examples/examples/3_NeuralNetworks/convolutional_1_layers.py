#!/usr/bin/env python

'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)
teX = teX.reshape(-1, 28, 28, 1)

# Parameters
learning_rate = 0.001
training_iters = 20
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout1 = 0.8 # Dropout, probability to keep units
dropout2 = 0.5 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 28,28,1])
y = tf.placeholder(tf.float32, [None, n_classes])
p_keep_conv = tf.placeholder(tf.float32) #dropout (keep probability)
p_keep_hidden = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, p_keep_conv, p_keep_hidden):
    # Reshape input picture
    #x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    #conv1 = tf.nn.dropout(conv1, p_keep_conv)

    # Convolution Layer
    #conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    #conv2 = maxpool2d(conv2, k=2)
    #conv2 = tf.nn.dropout(conv2, p_keep_conv)

    # Convolution Layer
    #conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    #conv3 = maxpool2d(conv3, k=2)
    #conv3 = tf.nn.dropout(conv3, p_keep_conv)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv1, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, p_keep_hidden)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)),

    # 5x5 conv, 32 inputs, 64 outputs
    #'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)),
    # 5x5 conv, 64 inputs, 128 outputs
    #'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    #'wd1': tf.Variable(tf.random_normal([4*4*128, 625], stddev=0.01)),
    #'wd1': tf.Variable(tf.random_normal([7*7*64, 625], stddev=0.01)),

    'wd1': tf.Variable(tf.random_normal([14*14*32, 625], stddev=0.01)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([625, n_classes], stddev=0.01))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    #'bc2': tf.Variable(tf.random_normal([64])),
    #'bc3': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([625])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, p_keep_conv, p_keep_hidden)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Keep training until reach max iterations
    for i in range(training_iters):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: trX[start:end], 
		                               y: trY[start:end],
                                       p_keep_conv: dropout1, 
									   p_keep_hidden: dropout2})

        # Calculate batch loss and accuracy
        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:256]
        loss, acc = sess.run([cost, accuracy], 
		                     feed_dict={x: teX[test_indices], 
							            y: teY[test_indices],
                                        p_keep_conv: 1.0, p_keep_hidden: 1.0})
        print("Iter " + str(i) + \
                  ", Loss= " + "{:.6f}".format(loss) + \
                  ", Training Accuracy= " + "{:.5f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for mnist test images
    for start, end in zip(range(0, len(teX), 128), range(128, len(teX), 128)):
        print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: teX[start:end],
                                      y: teY[start:end],
                                      p_keep_conv: 1.0, p_keep_hidden: 1.0}))



