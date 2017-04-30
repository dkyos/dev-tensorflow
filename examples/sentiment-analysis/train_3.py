#! /usr/bin/env python

import random
import os
import sys
import traceback

import numpy as np
import tensorflow as tf
from konlpy.tag import Twitter

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

TRAIN_FILENAME = 'ratings_train.txt'
TRAIN_DATA_FILENAME = TRAIN_FILENAME + '.data'
TEST_FILENAME = 'ratings_test.txt'
TEST_DATA_FILENAME = TEST_FILENAME + '.data'

#max_features = 55826
max_features = 56000
maxlen = 100  # cut texts after this number of words 
batch_size = 32

pos_tagger = Twitter()
vocab = dict()

def __FUNC__():
    return traceback.extract_stack(None, 2)[0][2]

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

def read_raw_data(filename, debug=False):
    with open(filename, 'r', encoding='utf-8') as f:
        print('loading data')
        data = [line.split('\t') for line in f.read().splitlines()]
        if debug: print("--- ", __FUNC__(), "\n", data)

        print('pos tagging to token')
        data = [(tokenize(row[1]), int(row[2])) for row in data[1:]]
        if debug: print("--- ", __FUNC__(), "\n", data)
    return data

def build_vocab(tokens):
    print('building vocabulary')
    vocab['#UNKOWN'] = 0
    #vocab['#PAD'] = 1
    vocab['#PAD'] = 0
    for t in tokens:
        if t not in vocab:
            vocab[t] = len(vocab)
    return vocab

def get_token_id(token, vocab):
    if token in vocab:
        return vocab[token]
    else:
        0 # unkown

def build_input(data, vocab):

    def get_onehot(index, size):
        onehot = [0] * size
        onehot[index] = 1
        return onehot

    print('building input')
    x_result = []
    y_result = []
    for d in data:
        sequence = [get_token_id(t, vocab) for t in d[0]]
        x_result.append(sequence)
        y_result.append(d[1])
    return x_result, y_result

data1 = read_raw_data(TRAIN_FILENAME, debug=False)
tokens1 = [t for d in data1 for t in d[0]]
vocab1 = build_vocab(tokens1)
print(len(vocab1), 'vocab size')
x_train, y_train = build_input(data1, vocab1)

data2 = read_raw_data(TEST_FILENAME, debug=False)
tokens2 = [t for d in data2 for t in d[0]]
vocab2 = build_vocab(tokens2)
print(len(vocab2), 'vocab size')
x_test, y_test = build_input(data2, vocab2)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=10,
    validation_data=(x_test, y_test))

#score, acc = model.evaluate(x_test, y_test,
#                            batch_size=batch_size)
#print('Test score:', score)
#print('Test accuracy:', acc)






















