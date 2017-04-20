#!/usr/bin/env python

'''
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf
import numpy as np
import matplotlib
import os

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def MinMaxScaler(data):
    ''' Min Max Normalization

    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]

    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]

    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# train Parameters
timesteps = seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
num_layers = 3 
learing_rate = 0.01
train_epochs = 100

# Open, High, Low, Volume, Close
#xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = np.loadtxt('dk-data-02-stock_daily.csv', delimiter=',')
#xy = xy[::-1]  # reverse order (chronically ordered)
xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]]  # Close as label

# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    #print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# train/test split
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size

print("train_size:%d test_size:%d" % (train_size, test_size))

trainX = np.array(dataX[0:train_size])
trainY = np.array(dataY[0:train_size])

testX = np.array(dataX[train_size:len(dataX)])
testY = np.array(dataY[train_size:len(dataY)])

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
#cell = tf.contrib.rnn.BasicLSTMCell(
cell = tf.contrib.rnn.LSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
#cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
loss_summ = tf.summary.scalar("loss", loss)
# optimizer
optimizer = tf.train.AdamOptimizer(learing_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./logs", sess.graph_def)
    merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for epoch  in range(train_epochs):
        summary, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(epoch, step_loss))
        writer.add_summary(summary, epoch)  # Write summary

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse))

    # predictions
    y_o = 0; p_o = 0
    iter = 0
    success = 0
    for (y, p) in zip(testY.ravel().tolist()[:], test_predict.ravel().tolist()[:]):
        iter = iter + 1
        if (y_o - y) > 0:
            if (p_o - p) > 0:
                success = success+1
                print("%f/%f (%f/%f) : ---/--- (%d, %.2f)" % (y_o, y, p_o, p, success, success/iter))
            else:
                print("%f/%f (%f/%f) : ---/+++ (%d, %.2f)" % (y_o, y, p_o, p, success, success/iter))
        else:
            if (p_o - p) > 0:
                print("%f/%f (%f/%f) : +++/--- (%d, %.2f)" % (y_o, y, p_o, p, success, success/iter))
            else:
                success = success+1
                print("%f/%f (%f/%f) : +++/+++ (%d, %.2f)" % (y_o, y, p_o, p, success, success/iter))
        y_o = y
        p_o = p 

    # Plot predictions
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()
