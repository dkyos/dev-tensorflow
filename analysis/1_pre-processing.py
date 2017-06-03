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
tf.flags.DEFINE_string("dst", "result.csv", "Original data file")
tf.flags.DEFINE_integer("start", 2000, "start year")
tf.flags.DEFINE_integer("end", 2021, "end year")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

url = FLAGS.src;
df = pd.read_csv(url, sep='|')

df['purchase'] = 0
df['disposal'] = 0
for i in range(FLAGS.start, FLAGS.end):
    df[str(i)] = 0

for index, row in df.iterrows():
    start = datetime.datetime.strptime(str(row["취득일자"]), "%Y%m%d").year
    df.set_value(index, 'purchase', start);

    end_str = str(row["처분일자"]);
    if end_str != "nan":
        end = datetime.datetime.strptime(str(int(float(end_str))), "%Y%m%d").year
        df.set_value(index, 'disposal', end);
    else:
        end = FLAGS.end;

    if start >= 2000: 
        for i in range(start, end+1):
            df.set_value(index, str(i), i - start + 1);
        for i in range(end, FLAGS.end + 1):
            df.set_value(index, str(i), end - i-1);

df.to_csv(FLAGS.dst, sep='|', index=False)

