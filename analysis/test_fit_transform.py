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


import pandas as pd
df = pd.DataFrame([
        ['서울','대위','175','75','1중대'],
        ['부산','중위','165','85','2중대'],
        ['부산','중위','165','85','2중대'],
        ['부산','중위','165','85','2중대'],
        [None,'중위','165','85','2중대'],
        ['성남','소위','180','70','1중대']        
])
df.columns = ['고향','계급','키', '몸무게','소속']
print ("================")
print (df)

df['고향'] = df['고향'].factorize()[0]
print ("================")
print (df)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelenc = LabelEncoder()
df['고향'] = labelenc.fit_transform(df['고향'].values)
df['계급'] = labelenc.fit_transform(df['계급'].values)
df['소속'] = labelenc.fit_transform(df['소속'].values)
print ("================")
print (df)
onehotencoder = OneHotEncoder(categorical_features=[0])
df = onehotencoder.fit_transform(df).toarray()   
print ("================")
print (df)

onehotencoder = OneHotEncoder(categorical_features='all')
df = onehotencoder.fit_transform(df).toarray()   
print ("================")
print (df)
