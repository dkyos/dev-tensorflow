#!/usr/bin/env python

# http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import keras.callbacks as cb
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model

time_steps = seq_length = 7
data_dim = 5

# Open,High,Low,Close,Volume
xy = np.loadtxt('dk-data-02-stock_daily.csv', delimiter=',')
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

# Keras Model for RNN:LSTM
model = Sequential()
model.add(LSTM(input_dim=data_dim, output_dim=64, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(128, return_sequences=False))
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
           epochs=100,
           validation_split=0.1,
           callbacks=callback_list)

# make predictions
test_predict = model.predict(testX)

# inverse values
# test_predict = scaler.transform(testPredict)
# testY = scaler.transform(testY)

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

# print(test_predict)
plt.plot(testY)
plt.plot(test_predict)
plt.show()
