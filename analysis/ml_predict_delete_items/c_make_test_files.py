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
import tensorflow as tf
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

# parameters
tf.flags.DEFINE_string("src", "test.csv", "Original data file")
tf.flags.DEFINE_string("train", "train.csv", "Train destination file")
tf.flags.DEFINE_string("predict", "predict.csv", "Predict destination file")

tf.flags.DEFINE_integer("train_year", 2015, "Train year")
tf.flags.DEFINE_integer("predict_year", 2016, "Predict year")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

fw_train = open(FLAGS.train, "w");
fw_predict = open(FLAGS.predict, "w");
writer_train = csv.writer(fw_train, delimiter='|');
writer_predict = csv.writer(fw_predict, delimiter='|');

url = FLAGS.src;
df = pd.read_csv(url, sep='|')

print ("### make train data")
df1 = df.loc[ (df[str(FLAGS.train_year)] == -1) ];
df2 = df.loc[ (df[str(FLAGS.train_year)] > 1) ];
print (df1.shape);
print (df2.shape);
frames = [df1, df2]
df_row = pd.concat(frames)

for index, row in df_row.iterrows():
    part = str(row["기관명"]).split('-')[0];
    due = int(row["내용연수"]);
    start = datetime.datetime.strptime(str(row["취득일자"]), "%Y%m%d").year

    life = int(row[str(FLAGS.train_year-1)]);
    k = int(row[str(FLAGS.train_year)]);
    if k > 0:
        y = 0
    elif k == -1:
        y = 1
    else:
        print ("BUG2 ---------------------")
        sys.exit()

    if (index % 1000)  == 0:
        print ("%s %d %d => %d" % (part, due, life, y))

    writer_train.writerow([part, due, life, y]);

print ("### make predict data")
df1 = df.loc[ (df[str(FLAGS.predict_year)] == -1) ];
df2 = df.loc[ (df[str(FLAGS.predict_year)] > 1) ];
print (df1.shape);
print (df2.shape);
frames = [df1, df2]
df_row = pd.concat(frames)

for index, row in df_row.iterrows():
    part = str(row["기관명"]).split('-')[0];
    due = int(row["내용연수"]);
    start = datetime.datetime.strptime(str(row["취득일자"]), "%Y%m%d").year

    life = int(row[str(FLAGS.predict_year-1)]);
    k = int(row[str(FLAGS.predict_year)]);
    if k > 0:
        y = 0
    elif k == -1:
        y = 1
    else:
        print ("BUG ---------------------")
        sys.exit()

    if (index % 1000)  == 0:
        print ("%s %d %d => %d" % (part, due, life, y))

    writer_predict.writerow([part, due, life, y]);

fw_train.close()
fw_predict.close()
