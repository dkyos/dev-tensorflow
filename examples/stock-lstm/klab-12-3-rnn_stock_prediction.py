#!/usr/bin/env python

import os
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import keras.callbacks as cb
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model

time_steps = seq_length = 100
data_dim = 4

tf.set_random_seed(777)  # reproducibility

# Open,High,Low,Close,Volume
xy = np.loadtxt('dk2-data-02-stock_daily.csv', delimiter=',')
#xy = xy[::-1]  # reverse order (chronically ordered)

# very important. It does not work without it.
scaler = MinMaxScaler(feature_range=(0, 1))
xy = scaler.fit_transform(xy)

x = xy
y = xy[:, [-1]]  # Close as label

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    #print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# split to train and testing
train_size = int(len(dataY) * 0.9)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    #predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs



# Keras Model for RNN:LSTM
model = Sequential()
model.add(LSTM(input_dim=data_dim, output_dim=50, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(output_dim=1))
model.add(Activation("linear"))
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

print(trainX.shape, trainY.shape)

callback_list= [
    cb.TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True),
    EarlyStopping(monitor='val_loss', patience=2)
]

# Train
model.fit( trainX, trainY,
           batch_size=128,
           epochs=10,
           validation_split=0.1,
           callbacks=callback_list)

# make predictions
test_predict = predict_point_by_point(model, testX)
plot_results(test_predict, testY)

predictions = predict_sequences_multiple(model, testX, time_steps, 7)
plot_results_multiple(predictions, testY, 7)

# inverse values
#test_predict = scaler.transform(test_predict)
#testY = scaler.transform(testY)

# predictions
y_o = 0; p_o = 0
iter = 0
success = 0
for (y, p) in zip(testY.ravel().tolist()[:], test_predict.ravel().tolist()[:]):
    iter = iter + 1
    if (y_o - y) > 0:
        if (p_o - p) > 0:
            success = success+1
            #print("%f/%f (%f/%f) : ---/--- (%d, %.2f)" % (y_o, y, p_o, p, success, success/iter))
        else:
            success
            #print("%f/%f (%f/%f) : ---/+++ (%d, %.2f)" % (y_o, y, p_o, p, success, success/iter))
    else:
        if (p_o - p) > 0:
            success
            #print("%f/%f (%f/%f) : +++/--- (%d, %.2f)" % (y_o, y, p_o, p, success, success/iter))
        else:
            success = success+1
            #print("%f/%f (%f/%f) : +++/+++ (%d, %.2f)" % (y_o, y, p_o, p, success, success/iter))
    y_o = y
    p_o = p 
        
print("(%d/%d, %.2f)" % (success, iter, success/iter))
