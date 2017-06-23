#!/usr/bin/env python

import spacy
import numpy as np;
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns

nlp = spacy.load('en')

print ("=== Load Data ========================")
df_train = pd.read_csv('../input/train.csv')
#print('Total number of question pairs for training: {}'.format(len(df_train)))
#df_test = pd.read_csv('../input/test.csv')
#print('Total number of question pairs for testing: {}'.format(len(df_test)))

def spacy_similarity(row):

    doc1 = nlp(row['question1'].lower());
    doc2 = nlp(row['question2'].lower());

    R = doc1.similarity(doc2);

    if row['is_duplicate'] == 1 :
        print ("1: %.2f" % (R) )
    else:
        print ("0: %.2f" % (R) )

    if R > 0.9:
        return 1
    else: 
        return 0

print ("=== spacy similarity match ========================")
train_similarity = df_train.apply(spacy_similarity, axis=1, raw=True)

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
print ("=== Precision score ========================")
print(precision_score(df_train['is_duplicate'], train_similarity))
print ("=== Classification Report ========================")
print(classification_report(df_train['is_duplicate'], train_similarity))

