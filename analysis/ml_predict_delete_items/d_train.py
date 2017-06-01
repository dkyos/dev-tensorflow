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
X[:, 0] = labelenc.fit_transform(X[:, 0])
print (X)
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()   
print (X)

# Split-out validation dataset
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

'''
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = ("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))
    print(msg)


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
'''

# Make predictions on validation dataset
#ml_alg = LogisticRegression();
ml_alg = LinearDiscriminantAnalysis();
#ml_alg = KNeighborsClassifier();
#ml_alg = DecisionTreeClassifier();
#ml_alg = SVC();
ml_alg.fit(X_train, Y_train)

predictions = ml_alg.predict(X_validation)
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
print(dataset.groupby('class').size())

# make X, Y
X = dataset.iloc[:, 0:3].values
Y = dataset.iloc[:, 3].values

# One Hot Encoding
print (X)
X[:, 0] = labelenc.fit_transform(X[:, 0])
print (X)
X = onehotencoder.fit_transform(X).toarray()   
print (X)

# Predict
predictions = ml_alg.predict(X)
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
