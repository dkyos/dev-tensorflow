#!/usr/bin/env python

# Check the versions of libraries

import csv

# Python version
import sys
#print('Python: {}'.format(sys.version))
# scipy
import scipy
#print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
#print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
#print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
#print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
#print('sklearn: {}'.format(sklearn.__version__))

# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

D_ORIGIN  = "02-D_origin.csv";
D_TOTAL   = "02-D_totoal.csv";
D_EXPIRED = "02-D_expired.csv";

fr = open(D_ORIGIN, "r");
read = csv.reader(fr);

fw_total = open(D_TOTAL, "w");
fw_expired = open(D_EXPIRED, "w");
writer_total = csv.writer(fw_total, delimiter=',');
writer_expired = csv.writer(fw_expired, delimiter=',');

row_num = 0;
for row in read:

    #product = "%s:%s" % (row[5], row[7])
    product = "%s" % (row[5])
    product.replace(" ", "")

    start = int(float(row[10])/10000);
    due = int(row[13]);

    if row[15]:
        end  = int(float(row[15])/10000);
        life = end - start;
    else:
        end = 0;
        life = 0;

    #if product != "데스크톱컴퓨터":
    #    continue;
    if product != "LCD패널또는모니터":
        continue;

    if row[15]:
        #print ("Expired,%s,%d,%d,%d,%d" % (product, start, end, due,life));
        writer_total.writerow(['Expired', product, start, end, due,life]);
        writer_expired.writerow([start, due,life]);
    else:
        #print ("Used,%s,%d,%d,%d,%d" % (product, start, end, due, life));
        writer_total.writerow(['Used', product, start, end, due,life]);

fw_total.close()
fw_expired.close()

# Load dataset
url = D_TOTAL;
#class = life
names = ['expired','product', 'start', 'end', 'due', 'life']
dataset = pandas.read_csv(url, delimiter=',', names=names)

#for i in range(2000, 2020):
#    dataset.insert(6, i, 0)

# 2000 ~ 2020
data_new = [0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0]
data_cur = [0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0]
data_del= [0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0]

for i, row in enumerate(dataset.values):
    print("=====================")
    print("year: %d -- %d" % (row[2], row[3]));

    if row[2] > 2000:
        start = row[2] - 2000;
    else:
        start = 2000 - 2000;

    if row[3] > 2000:
        end = row[3] - 2000;
    else:
        end = 2020 - 2000;

    print("index: %d -- %d" % (start, end));

    data_new[start] = data_new[start] + 1; 

    for i in range(start, end, 1):
           data_cur[i] = data_cur[i] + 1; 

    data_del[end] = data_del[end] + 1; 

    #print(row)
    #expired, product, start, end, due = row
    print ("[new    ] ", (['%2d'% data_new[n] for n in range(len(data_new))]));
    print ("[current] ", (['%2d'% data_cur[n] for n in range(len(data_cur))]));
    print ("[del    ] ", (['%2d'% data_del[n] for n in range(len(data_del))]));

# class distribution
print(dataset.groupby('start').size())
print(dataset.groupby('end').size())


