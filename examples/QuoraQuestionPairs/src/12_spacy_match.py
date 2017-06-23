#!/usr/bin/env python

import spacy
import numpy as np;
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score

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

    '''
    if row['is_duplicate'] == 1 :
        print ("1: %.2f" % (R) )
    else:
        print ("0: %.2f" % (R) )
    '''

    return R

print ("=== spacy similarity match ========================")
train_similarity = df_train.apply(spacy_similarity, axis=1, raw=True)


#print (train_similarity.head())
print (train_similarity[:20])

df_train_1 = df_train[ df_train['is_duplicate'] == 1 ]
#print (df_train_1[:20])

train_similarity_0 = train_similarity[ df_train['is_duplicate'] == 0 ]
train_similarity_1 = train_similarity[ df_train['is_duplicate'] == 1 ]
print (train_similarity_0[:10])
print (train_similarity_1[:10])

pal = sns.color_palette();

plt.figure(figsize=(15, 10))
plt.hist(train_similarity_0, bins=50, color=pal[2], normed=True)
plt.title('Normalised histogram of spacy similarity score (with 0)', fontsize=15)
plt.legend()
plt.xlabel('Number', fontsize=15)
plt.ylabel('Similarity', fontsize=15)
plt.show()

plt.figure(figsize=(15, 10))
plt.hist(train_similarity_1, bins=50, color=pal[1], normed=True, label='train')
plt.title('Normalised histogram of spacy similarity score (with 1)', fontsize=15)
plt.legend()
plt.xlabel('Number', fontsize=15)
plt.ylabel('Similarity', fontsize=15)
plt.show()

#print ("=== Precision score ========================")
#print(precision_score(df_train['is_duplicate'], train_similarity))
#print ("=== Classification Report ========================")
#print(classification_report(df_train['is_duplicate'], train_similarity))

