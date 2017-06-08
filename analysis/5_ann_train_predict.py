#!/usr/bin/env python

### libraries
import csv
import sys
import scipy
import numpy 
import numpy as np
import matplotlib
import sklearn
import pandas
import datetime
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('sklearn: {}'.format(sklearn.__version__))

# parameters
tf.flags.DEFINE_string("train", "train.csv", "Train file path with name")
tf.flags.DEFINE_string("predict", "predict.csv", "Predict file path with name")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

np.random.seed(777)

##############################################################################
## Train

print ("############### Train ####################")

# Load dataset
url = FLAGS.train;
names = ['part', 'due', 'life', 'class']
df = pandas.read_csv(url, delimiter='|', names=names)

# check data
print(df.shape)
print(df.head(10))
print(df.describe())
print(df.groupby('class').size())

# reduce data with 'class' == 0
dataset = df.drop( df[df['class'] == 0].sample(frac=.8).index )

# check data
print(dataset.groupby('class').size())

# make X, Y
X = dataset.iloc[:, 0:3].values
Y = dataset.iloc[:, 3].values

# One Hot Encoding 
labelenc = LabelEncoder()
print (X)
labelenc.fit( ['A', 'B','C','D','E','F','G','H','I','J', \
'K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z', \
'AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ',\
'AK','AL','AM','AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ'] )
#labelenc.fit(X[:, 0])
print (labelenc.classes_)
X[:, 0] = labelenc.transform(X[:, 0])
print (X)
onehotencoder = OneHotEncoder(categorical_features=[0])
onehotencoder.fit(X)
print (onehotencoder.n_values_)
X = onehotencoder.transform(X).toarray()
print (onehotencoder.active_features_)
print (onehotencoder.feature_indices_)
print (X)

# Split-out validation dataset
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden Layer
classifier.add(Dense(units=50, kernel_initializer='glorot_uniform', activation='relu', input_dim=52))
classifier.add(Dense(units=50, kernel_initializer='glorot_uniform', activation='relu'))
classifier.add(Dense(units=50, kernel_initializer='glorot_uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='random_normal', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
print ("fit ====================")
classifier.fit(X_train, Y_train, batch_size=10, epochs=11)

# Predicting the Test set results
print ("predict ====================")
predictions = classifier.predict(X_validation)
predictions = (predictions > 0.5 )

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# check
o_o = 0; o_l = 0; l_o = 0; l_l = 0
for i, j in zip(Y_validation, predictions): 
    #print("(%d, %d)" % (i, j))
    if (i == 0 and j == 0):
        o_o = o_o + 1;
    elif (i == 0 and j == 1):
        o_l = o_l + 1;

    if (i == 1 and j == 0):
        l_o = l_o + 1;
    elif (i == 1 and j == 1):
        l_l = l_l + 1;

print ("(0 => 1) %d" % o_l)
print ("(0 => 0) %d %3.2f (%d/%d)" % (o_o, float(o_o/(o_o + o_l)), o_o, o_o + o_l))
print ("(1 => 0) %d" % l_o)
print ("(1 => 1) %d %3.2f (%d/%d)" % (l_l, float(l_l/(l_o + l_l)), l_l, l_o + l_l))

print (" => real %3.2f" % ( (o_l + l_l)/(l_o + l_l) ) )

##############################################################################
## Predict

print ("############### Predict ####################")

# Load dataset
url = FLAGS.predict
names = ['part', 'due', 'life', 'class']
dataset = pandas.read_csv(url, delimiter='|', names=names)

# check data
print(dataset.groupby('part').size().sort_values(ascending=False))
print(dataset.groupby('due').size().sort_values(ascending=False))
print(dataset.groupby('life').size().sort_values(ascending=False))
print(dataset.groupby('class').size().sort_values(ascending=False))

# make X, Y
X = dataset.iloc[:, 0:3].values
Y = dataset.iloc[:, 3].values

# One Hot Encoding
print (X)
X[:, 0] = labelenc.transform(X[:, 0])
print (X)
X = onehotencoder.transform(X).toarray()
print (X)

# Predict
predictions = classifier.predict(X)
predictions = (predictions > 0.5 )

print(accuracy_score(Y, predictions))
print(confusion_matrix(Y, predictions))
print(classification_report(Y, predictions))

# check
o_o = 0; o_l = 0; l_o = 0; l_l = 0
for i, j in zip(Y, predictions): 
    #print("(%d, %d)" % (i, j))
    if (i == 0 and j == 0):
        o_o = o_o + 1;
    elif (i == 0 and j == 1):
        o_l = o_l + 1;

    if (i == 1 and j == 0):
        l_o = l_o + 1;
    elif (i == 1 and j == 1):
        l_l = l_l + 1;

print ("(0 => 1) %d" % o_l)
print ("(0 => 0) %d %3.2f (%d/%d)" % (o_o, float(o_o/(o_o + o_l)), o_o, o_o + o_l))
print ("(1 => 0) %d" % l_o)
print ("(1 => 1) %d %3.2f (%d/%d)" % (l_l, float(l_l/(l_o + l_l)), l_l, l_o + l_l))

print (" => real %3.2f" % ( (o_l + l_l)/(l_o + l_l) ) )
