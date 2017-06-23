#!/usr/bin/env python

import spacy
import numpy as np;
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn

print ("=== Load Data ========================")
df_train = pd.read_csv('../input/train.csv')
#print('Total number of question pairs for training: {}'.format(len(df_train)))
#df_test = pd.read_csv('../input/test.csv')
#print('Total number of question pairs for testing: {}'.format(len(df_test)))

def path_similarity_won(synset, ss):
    a = synset.path_similarity(ss);
    if a is not None:
        ret = a;
    else:
        #ret = float('-inf')
        ret = 0
    return ret;

""" Convert between a Penn Treebank tag to a simplified Wordnet tag """;
def penn_to_wn(tag):
    if tag.startswith('N'):
        return 'n';

    if tag.startswith('V'):
        return 'v';

    if tag.startswith('J'):
        return 'a';

    if tag.startswith('R'):
        return 'r';

    return None;

def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None;

    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None;

""" compute the sentence similarity using Wordnet """;
def sentence_similarity(sentence1, sentence2):
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = max([path_similarity_won(synset, ss) for ss in synsets2])

        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
        count += 1

    # Average the values
    score /= count
    return score

""" compute the symmetric sentence similarity using Wordnet """;
def symmetric_sentence_similarity(sentence1, sentence2):
    return (sentence_similarity(sentence1, sentence2) + sentence_similarity(sentence2, sentence1)) / 2 
         
nlp = spacy.load('en')

def wordnet_similarity(row):

    #doc1 = nlp(row['question1'].lower());
    #doc2 = nlp(row['question2'].lower());

    doc1 = row['question1'].lower()
    doc2 = row['question2'].lower()

    R = symmetric_sentence_similarity(doc1, doc2)

    if row['is_duplicate'] == 1 :
        print ("1: %.2f" % (R) )
    else:
        print ("0: %.2f" % (R) )

    if R > 0.9:
        return 1
    else: 
        return 0

print ("=== spacy similarity match ========================")
train_similarity = df_train.apply(wordnet_similarity, axis=1, raw=True)

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
print ("=== Precision score ========================")
print(precision_score(df_train['is_duplicate'], train_similarity))
print ("=== Classification Report ========================")
print(classification_report(df_train['is_duplicate'], train_similarity))

