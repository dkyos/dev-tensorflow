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

FILE_SRC  = "test.csv";
FILE_DST  = "result.csv";

START = 2000;
END   = 2021;

url = FILE_SRC;
df = pd.read_csv(url, sep='|')

for i in range(START, END):
    df[str(i)] = 0

for index, row in df.iterrows():
    start = datetime.datetime.strptime(str(row["취득일자"]), "%Y%m%d").year
    end_str = str(row["처분일자"]);
    if end_str != "nan":
        end = datetime.datetime.strptime(str(int(float(end_str))), "%Y%m%d").year
    else:
        end = END;

    #print("---------")
    #print ("%d,%d" % (start, end));

    if start >= 2000: 
        for i in range(start, end+1):
            df.set_value(index, str(i), i - start + 1);
        for i in range(end, END+1):
            df.set_value(index, str(i), end - i-1);

df.to_csv(FILE_DST, sep='|', index=False)

