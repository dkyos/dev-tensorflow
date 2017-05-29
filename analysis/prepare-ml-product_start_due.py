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

D_ORIGIN_FILE  = "02_20170516_G.csv";
D_TOTAL   = "02-G_totoal.csv";
D_EXPIRED = "02-G_expired.csv";

fw_total = open(D_TOTAL, "w");
fw_expired = open(D_EXPIRED, "w");
writer_total = csv.writer(fw_total, delimiter='|');
writer_expired = csv.writer(fw_expired, delimiter='|');

# Load dataset
url = D_ORIGIN_FILE;
df = pd.read_csv(url, sep='|')

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
        writer_expired.writerow([product, start, due, life]);
    else:
        print ("Used,%s,%d,%d,%d,%d" % (product, start, end, due, life));
        writer_total.writerow(['Used', product, start, end, due,life]);

fw_total.close()
fw_expired.close()

# Load dataset
url = D_EXPIRED;
#class = life
names = ['product', 'start', 'due', 'class']
dataset = pandas.read_csv(url, delimiter='|', names=names)

# shape
print(dataset.shape)
# head
print(dataset.head(10))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())

'''
# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
dataset.plot(kind='box')
plt.show()

# histograms
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()
'''

# Split-out validation dataset
dataset['product'] = dataset['product'].factorize()[0]
X = dataset.iloc[:, 0:3].values
Y = dataset.iloc[:, 3].values

# Encodeing Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelenc = LabelEncoder()
X[:, 0] = labelenc.fit_transform(X[:, 0])
X[:, 1] = labelenc.fit_transform(X[:, 1])
print (X)
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()   
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()   
print (X)

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = ("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))
    print(msg)


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Make predictions on validation dataset
#knn = KNeighborsClassifier()
knn = DecisionTreeClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
