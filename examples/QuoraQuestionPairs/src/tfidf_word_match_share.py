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


train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
#test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

from nltk.corpus import stopwords
stops = set(stopwords.words("english"))

from collections import Counter

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

def tfidf_word_match_share(row):
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

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)

    if R > 0.5:
        return 1
    else: 
        return 0

tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)

#from sklearn.metrics import roc_auc_score
#print(roc_auc_score(df_train['is_duplicate'], tfidf_train_word_match.fillna(0)))

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
print ("=== Precision score ========================")
print(precision_score(df_train['is_duplicate'], tfidf_train_word_match ))
print ("=== Classification Report ========================")
print(classification_report(df_train['is_duplicate'], tfidf_train_word_match ))


