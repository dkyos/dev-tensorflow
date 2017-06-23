#!/usr/bin/env python

import numpy as np;
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns

print ("=== Load Data ========================")
df_train = pd.read_csv('../input/train.csv')
#print('Total number of question pairs for training: {}'.format(len(df_train)))
#df_test = pd.read_csv('../input/test.csv')
#print('Total number of question pairs for testing: {}'.format(len(df_test)))

# stopwords란 의미를 가지지 않는 단어들: 관사, 조사, 전치사, 접속사 등
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))

    if R > 0.5:
        return 1
    else: 
        return 0

print ("=== word match share ========================")
train_word_match = df_train.apply(word_match_share, axis=1, raw=True)
#print ( train_word_match.head())

#from sklearn.metrics import roc_auc_score
#print(roc_auc_score(df_train['is_duplicate'], train_word_match))

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
print ("=== Precision score ========================")
print(precision_score(df_train['is_duplicate'], train_word_match))
print ("=== Classification Report ========================")
print(classification_report(df_train['is_duplicate'], train_word_match))

