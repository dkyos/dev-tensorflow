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

D_ORIGIN_FILE  = "1_plan_20170526.csv"

# Load dataset
url = D_ORIGIN_FILE;
df = pd.read_csv(url, sep='|')


multikey_dic = {}
year_value = {}

#>>> alias1 in mydict
#True

for row in df.itertuples(index=True, name='Pandas'):

    #기관명|분류번호|품명|취득계획수량| 취득계획금액 |확정일자|차수|연도|비고

    name = str(getattr(row, "기관명")).replace(" ", "")
    product = str(getattr(row, "품명")).replace(" ", "")
    year = int(str(getattr(row, "연도")).replace(" ", ""))

    update = int(str(getattr(row, "차수")).replace(" ", ""))
    plan = int(str(getattr(row, "취득계획수량")).replace(" ", ""))

    if product != "데스크톱컴퓨터": 
    #if product != "LCD패널또는모니터":
         continue;

    alias = (name, product, year)

    if alias in multikey_dic:
        #print (multikey_dic[alias])
        a = multikey_dic[alias]
        if( update > a[0]):
            multikey_dic[alias] = [update, plan]
            #print ("Update: [%d, %d] => [%d, %d]" % (a[0], a[1], update, plan))
    else:
        multikey_dic[alias] = [update, plan]

for key in multikey_dic.keys():
    a = multikey_dic[key]
    year = key[2]

    print ("=================")
    print (key)
    print (a)

    alias = (year)
    if alias in year_value:
        year_value[alias] = year_value[alias] + a[1];
    else:
        year_value[alias] = a[1];

    for key in year_value.keys():
        a = year_value[key]
        print ("%d, %d" % (key, a))





