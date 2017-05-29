#!/usr/bin/env python

### libraries
import csv
import sys
import scipy
import numpy
import matplotlib
import sklearn
import pandas
import pandas as pd
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
print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('sklearn: {}'.format(sklearn.__version__))

D_ORIGIN_FILE  = "concat_result.csv";
D_TOTAL   = "02-D_totoal.csv";
D_EXPIRED = "02-D_expired.csv";

fw_total = open(D_TOTAL, "w");
fw_expired = open(D_EXPIRED, "w");
writer_total = csv.writer(fw_total, delimiter='|');
writer_expired = csv.writer(fw_expired, delimiter='|');

# Load dataset
url = D_ORIGIN_FILE;
df = pd.read_csv(url, delimiter='|')

for row in df.itertuples(index=True, name='Pandas'):

    product = str(getattr(row, "품명")).replace(" ", "")
    #if product != "데스크톱컴퓨터": 
    #    continue;

    start = int(float(getattr(row, "취득일자"))/10000);
    due = int(getattr(row, "내용연수"));

    end_str = str(getattr(row, "처분일자"));
    if end_str != "nan":
        end  = int(float(end_str)/10000);
        life = end - start;
    else:
        end = 0;
        life = 0;

    if end_str:
        print ("Expired,%s,%d,%d,%d,%d" % (product, start, end, due,life));
        writer_total.writerow(['Expired', product, start, end, due,life]);
        writer_expired.writerow([start, due, life]);
    else:
        print ("Used,%s,%d,%d,%d,%d" % (product, start, end, due, life));
        writer_total.writerow(['Used', product, start, end, due,life]);

fw_total.close()
fw_expired.close()

# Load dataset
url = D_TOTAL;
#class = life
names = ['expired','product', 'start', 'end', 'due', 'life']
dataset = pandas.read_csv(url, delimiter='|', names=names)

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
        data_new[start] = data_new[start] + 1; 
    else:
        start = 2000 - 2000;

    if row[3] > 2000:
        end = row[3] - 2000;
        data_del[end] = data_del[end] + 1; 
    else:
        end = 2020 - 2000;

    print("index: %d -- %d" % (start, end));

    for i in range(start, end, 1):
           data_cur[i] = data_cur[i] + 1; 

    #print(row)
    #expired, product, start, end, due = row
    print ("[new    ] ", (['%2d'% data_new[n] for n in range(len(data_new))]));
    print ("[current] ", (['%2d'% data_cur[n] for n in range(len(data_cur))]));
    print ("[del    ] ", (['%2d'% data_del[n] for n in range(len(data_del))]));

# class distribution
print(dataset.groupby('start').size())
print(dataset.groupby('end').size())


