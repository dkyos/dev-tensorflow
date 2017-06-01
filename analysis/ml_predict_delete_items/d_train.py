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
print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('sklearn: {}'.format(sklearn.__version__))

FILE_SRC        = "train.csv";
FILE_PREDICT    = "predict.csv";

##############################################################################
## Train

# Load dataset
url = FILE_SRC;
names = ['part', 'due', 'life', 'class']
df = pandas.read_csv(url, delimiter='|', names=names)

# check data
print(df.shape)
print(df.head(10))
print(df.describe())
print(df.groupby('class').size())

# reduce data with 'class' == 0
print(df[df['class'] == 0].sample(frac=.1).index )
dataset = df.drop( df[df['class'] == 0].sample(frac=.9).index )

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

# Make predictions on validation dataset
ml_alg = LogisticRegression();
ml_alg.fit(X_train, Y_train)

predictions = ml_alg.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

##############################################################################
## Predict

# Load dataset
url = FILE_PREDICT;
names = ['part', 'due', 'life', 'class']
dataset = pandas.read_csv(url, delimiter='|', names=names)

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
predictions = knn.predict(X)
print(accuracy_score(Y, predictions))
print(confusion_matrix(Y, predictions))
print(classification_report(Y, predictions))

# check
for i, j in zip(Y, predictions): 
    if i == 1 and i == j:
        print("OK")
    elif i == 1 and i != j:
        print("FAIL")

