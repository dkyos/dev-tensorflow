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
print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('sklearn: {}'.format(sklearn.__version__))

FILE_SRC  = "result.csv";

START = 2000;
END   = 2021;

url = FILE_SRC;
df = pd.read_csv(url, sep='|')

print ("===========================")
for i in range(START, END+1):
    print ("-- Retain -----------")
    print (i)
    print (df.loc[df[str(i)] >= 1].shape)

print ("===========================")
for i in range(START, END+1):
    print ("-- Delete -----------")
    print (i)
    print (df.loc[df[str(i)] == -1].shape)

print ("===========================")
for i in range(START, END+1):
    print ("-- Buy -----------")
    print (i)
    print (df.loc[df[str(i)] == 1].shape)

