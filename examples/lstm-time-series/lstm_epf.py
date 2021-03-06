#!/usr/bin/env python

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python import SKCompat #

from sklearn.metrics import mean_squared_error, mean_absolute_error
from lstm_predictior import generate_data, load_csvdata, lstm_model


LOG_DIR = 'resources/epf_logs'
TIMESTEPS = 10
RNN_LAYERS = [{'num_units': TIMESTEPS}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 100000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100

dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M')
rawdata = pd.read_csv("./RealMarketPriceDataPT.csv", 
                   parse_dates={'timeline': ['date', '(UTC)']}, 
                   index_col='timeline', date_parser=dateparse)


X, y = load_csvdata(rawdata, TIMESTEPS, seperate=False)

regressor = SKCompat(learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),model_dir=LOG_DIR)) # new
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'], every_n_steps=PRINT_STEPS, early_stopping_rounds=1000)

regressor.fit(X['train'], y['train'],
              monitors=[validation_monitor],
              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS)
print(X['test'].shape)
print(y['test'].shape)

predicted = regressor.predict(X['test']) # ,as_iterable=False)
score = mean_squared_error(predicted, y['test'])
print ("MSE: %f" % score)

plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(y['test'], label='test')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()

