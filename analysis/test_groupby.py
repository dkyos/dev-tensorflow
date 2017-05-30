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

## files
#D_ORIGIN_FILE  = "02_20170516_A.csv";
#D_ORIGIN_FILE  = "02_20170516_G.csv";
D_ORIGIN_FILE  = "00_total_concat_data.csv";

#############################
## read csv with pandas api
print ("===============")
table1 = pd.read_csv(D_ORIGIN_FILE, encoding='utf-8', sep='|')
print (table1.head(10))

#############################
## groupby top 10
print( table1.groupby('10품종').size().sort_values(ascending=False).head(10) )
print( table1.groupby('품명').size().sort_values(ascending=False).head(10) )

print( table1.groupby('취득일자').size().sort_values(ascending=False).head(10) )
print( table1.groupby('내용연수').size().sort_values(ascending=False).head(10) )
print( table1.groupby('처분일자').size().sort_values(ascending=False).head(10) )

#############################
## show korean in plot
#print (matplotlib.rcParams["font.family"])
import matplotlib.font_manager as fm
#font_location = "/usr/share/fonts/truetype/nanum/NanumGothic_Coding.ttf"
font_location = "/usr/share/fonts/truetype/nanum/NanumMyeongjoBold.ttf"
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)
matplotlib.rcParams.update({'font.weight': 'bold'})
#matplotlib.rcParams.update({'font.size': 14})

#############################
## show plot
table1.groupby('10품종').size().sort_values(ascending=False).head(10).plot(kind='bar',stacked=True)
plt.show()

table1.groupby('품명').size().sort_values(ascending=False).head(10).plot(kind='bar',stacked=True)
plt.show()

