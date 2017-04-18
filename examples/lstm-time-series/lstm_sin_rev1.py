#!/usr/bin/env python

### Libraries
#
# - numpy: package for scientific computing
# - pandas: data structures and data analysis tools
# - tensorflow: open source software library for machine intelligence
# - matplotlib: 2D plotting library
#
#
# - **learn**: Simplified interface for TensorFlow (mimicking Scikit Learn) for Deep Learning
# - mse: "mean squared error" as evaluation metric
# - **lstm_predictor**: our lstm class
#

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python import SKCompat #

from sklearn.metrics import mean_squared_error
from lstm_predictior import generate_data, lstm_model

### Parameter definitions
#
# - LOG_DIR: log file
# - TIMESTEPS: RNN time steps
# - RNN_LAYERS: RNN layer 정보
# - DENSE_LAYERS: DNN 크기 [10, 10]: Two dense layer with 10 hidden units
# - TRAINING_STEPS: 학습 스텝
# - BATCH_SIZE: 배치 학습 크기
# - PRINT_STEPS: 학습 과정 중간 출력 단계 (전체의 1% 해당하는 구간마다 출력)

LOG_DIR = 'resources/sin_logs/'
TIMESTEPS = 1
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 100
#BATCH_SIZE = 100
BATCH_SIZE = 10
PRINT_STEPS = TRAINING_STEPS / 10

### Create a regressor with TF Learn
#
# : 예측을 위한 모델 생성. TF learn 라이브러리에 제공되는 Estimator를 사용.
#
# **Parameters**:
#
# - model_fn: 학습 및 예측에 사용할 모델
#

regressor = SKCompat(learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),model_dir=LOG_DIR)) # new

### Generate a dataset
#
# 1. generate_data: 학습에 사용될 데이터를 특정 함수를 이용하여 만듭니다.
#  - fct: 데이터를 생성할 함수
#  - x: 함수값을 관측할 위치
#  - time_steps: 관측(observation)
#  - seperate: check multimodal
# 1. ValidationMonitor: training 이후, validation 과정을 모니터링
#  - x
#  - y
#  - every_n_steps: 중간 출력
#  - early_stopping_rounds

X, y = generate_data(np.sin, np.linspace(0, 100, 10000, dtype=np.float32), TIMESTEPS, seperate=False)
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'], every_n_steps=PRINT_STEPS, early_stopping_rounds=1000)
#print(X['train'])
#print(y['train'])

### Train and validation
#
# - fit: training data를 이용해 학습, 모니터링과 로그를 통해 기록
#
#

regressor.fit(X['train'], y['train'],
              monitors=[validation_monitor],
              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS)
print(X['test'].shape)
print(y['test'].shape)

# ## Evaluate using test set
#
# Evaluate our hypothesis using test set. The mean squared error (MSE) is used for the evaluation metric.
#
#
predicted = regressor.predict(X['test']) # ,as_iterable=False)
score = mean_squared_error(predicted, y['test'])
print ("MSE: %f" % score)

#idx=0
#for pred, label in zip(predicted, y['test']):
#    print("%d  Predction:%.3f Label:%.3f => %.3f " % (idx, pred, label, pred/label) )
#    idx=idx+1

plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(y['test'], label='test')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()
