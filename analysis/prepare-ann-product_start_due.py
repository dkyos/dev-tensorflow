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

D_ORIGIN_FILE  = "02-D_origin.csv";
D_TOTAL   = "02-D_totoal.csv";
D_EXPIRED = "02-D_expired.csv";

fw_total = open(D_TOTAL, "w");
fw_expired = open(D_EXPIRED, "w");
writer_total = csv.writer(fw_total, delimiter=',');
writer_expired = csv.writer(fw_expired, delimiter=',');

# Load dataset
url = D_ORIGIN_FILE;
df = pd.read_csv(url, delimiter=',')

for row in df.itertuples(index=True, name='Pandas'):

    product = str(getattr(row, "f")).replace(" ", "")
    type = str(getattr(row, "h")).replace(" ", "")
    #if product != "데스크톱컴퓨터": 
    #    continue;

    start = int(float(getattr(row, "k"))/10000);
    due = int(getattr(row, "n"));

    end_str = str(getattr(row, "p"));
    if end_str != "nan":
        end  = int(float(end_str)/10000);
        life = end - start;
    else:
        end = 0;
        life = 0;

    if end_str != "nan":
        print ("Expired,%s,%s,%d,%d,%d,%d" % (product, type, start, end, due,life));
        writer_total.writerow(['Expired', product, type, start, end, due,life]);
        writer_expired.writerow([product, type, start, due, life]);
    else:
        print ("Used,%s,%s,%d,%d,%d,%d" % (product, type, start, end, due, life));
        writer_total.writerow(['Used', product, type, start, end, due,life]);

fw_total.close()
fw_expired.close()

# Load dataset
url = D_EXPIRED;
#class = life
names = ['product', 'type', 'start', 'due', 'class']
dataset = pandas.read_csv(url, delimiter=',', names=names)

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
dataset['type'] = dataset['type'].factorize()[0]
X = dataset.iloc[:, 0:4].values
Y = dataset.iloc[:, 4].values

# Encodeing Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelenc = LabelEncoder()
X[:, 0] = labelenc.fit_transform(X[:, 0])
X[:, 1] = labelenc.fit_transform(X[:, 1])
#print ("======="); print (X.shape); print (X);
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()   
#print ("======="); print (X.shape); print (X);
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()   
#print ("======="); print (X.shape); print (X);

validation_size = 0.20
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print (y_train.shape)
print (y_test.shape)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden Layer
classifier.add(Dense(units=64, kernel_initializer='normal', activation='relu', input_dim=147))
classifier.add(Dense(units=32, kernel_initializer='normal', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='normal', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
print ("fit ====================")
classifier.fit(X_train, y_train, batch_size=10, epochs=1000)

# Predicting the Test set results
print ("predict ====================")
y_pred = classifier.predict(X_test)
print (y_pred)
y_pred = (y_pred > 0.5)
print ("predict ====================")
print (y_pred)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
print ("matrix ====================")
cm = confusion_matrix(y_test, y_pred)

