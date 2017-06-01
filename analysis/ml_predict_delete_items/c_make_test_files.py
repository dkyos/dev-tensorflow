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
TRAIN_DST  = "train.csv";
PREDICT_DST  = "predict.csv";

START = 2000;
END   = 2021;

TRAIN_YEAR = 2015
PREDICT_YEAR = 2016

fw_train = open(TRAIN_DST, "w");
fw_predict = open(PREDICT_DST, "w");
writer_train = csv.writer(fw_train, delimiter='|');
writer_predict = csv.writer(fw_predict, delimiter='|');

url = FILE_SRC;
df = pd.read_csv(url, sep='|')

print ("### make train data")
df1 = df.loc[ (df[str(TRAIN_YEAR)] == -1) ];
df2 = df.loc[ (df[str(TRAIN_YEAR)] > 1) ];
print (df1.shape);
print (df2.shape);
frames = [df1, df2]
df_row = pd.concat(frames)

for index, row in df_row.iterrows():
    part = str(row["기관명"]).split('-')[0];
    due = int(row["내용연수"]);
    start = datetime.datetime.strptime(str(row["취득일자"]), "%Y%m%d").year

    life = int(row[str(TRAIN_YEAR-1)]);
    k = int(row[str(TRAIN_YEAR)]);
    if k > 0:
        y = 0
    elif k == -1:
        y = 1
    else:
        print ("BUG2 ---------------------")
        sys.exit()

    print ("%s %d %d => %d" % (part, due, life, y))
    writer_train.writerow([part, due, life, y]);

print ("### make predict data")
df1 = df.loc[ (df[str(PREDICT_YEAR)] == -1) ];
df2 = df.loc[ (df[str(PREDICT_YEAR)] > 1) ];
print (df1.shape);
print (df2.shape);
frames = [df1, df2]
df_row = pd.concat(frames)

for index, row in df_row.iterrows():
    part = str(row["기관명"]).split('-')[0];
    due = int(row["내용연수"]);
    start = datetime.datetime.strptime(str(row["취득일자"]), "%Y%m%d").year

    life = int(row[str(PREDICT_YEAR-1)]);
    k = int(row[str(PREDICT_YEAR)]);
    if k > 0:
        y = 0
    elif k == -1:
        y = 1
    else:
        print ("BUG ---------------------")
        sys.exit()

    print ("%s %d %d => %d" % (part, due, life, y))
    writer_predict.writerow([part, due, life, y]);

fw_train.close()
fw_predict.close()

