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

D_ORIGIN_FILE  = "00_total_concat_data.csv";
D_FILTERED_FILE = "filtered_result.csv";

fw_filter = open(D_FILTERED_FILE, "w");
writer_filter = csv.writer(fw_filter, delimiter='|');

# Load dataset
url = D_ORIGIN_FILE;
df = pd.read_csv(url, sep='|')

for row in df.itertuples(index=True, name='Pandas'):
    product = str(getattr(row, "품명")).replace(" ", "")
    #if product != "데스크톱컴퓨터": 
    if product != "LCD패널또는모니터":
        continue;

    writer_filter.writerow(row);


fw_filter.close()


