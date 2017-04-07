#! /usr/bin/env python

from data import *
from cnn import TextCNN
import tensorflow as tf
import random
import numpy as np
import os
import sys
import time
import datetime
import pickle
import prepare_konlpy as prep
from konlpy.tag import Twitter

dev_sample_percentage = 0.1
num_epochs = 1
batch_size = 50

TRAIN_FILENAME = 'ratings_train.txt'
TRAIN_PICKLE= 'pickle.data'

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    print("============================================")
    print(" Generates a batch iterator for a dataset.")

    np.random.seed(10)
    shuffle=True

    data = np.array(data)
    data_size = len(data)

    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    print("data size: %d, # of batches per epoch: %d " % (data_size, num_batches_per_epoch))

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def train():

    #########################################################################
    if os.path.exists(TRAIN_PICKLE):
        with open(TRAIN_PICKLE, 'rb') as f:  # Python 3: open(..., 'rb')
            train_docs, train_labels = pickle.load(f)
    else:
        train_data = prep.read_data(TRAIN_FILENAME)
        #print(train_data)

        pos_tagger = Twitter()

        start_time = time.time()
        train_docs = [prep.tokenize(pos_tagger, row[1]) for row in train_data]
        #print(train_docs)
        train_labels = [float(row[2]) for row in train_data]
        #print(train_labels)
        print('---- %s seconds elapsed ----' % (time.time() - start_time))

        # Saving the objects:
        with open(TRAIN_PICKLE, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([train_docs, train_labels], f)

    #########################################################################
    train_labels = prep.labeller(train_labels)
    #print(train_labels)
    train_docs_p = prep.pad_sentences(train_docs)
    #print(train_docs_p)

    # Build vocabulary
    voc, voc_inv = prep.build_vocab(train_docs_p)
    #print(voc)
    #print(voc_inv)

    # Write vocabulary
    with open('vocab.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([voc, voc_inv], f)

    x, y = prep.build_input_data(train_docs_p, train_labels, voc)
    #print(x)
    #print(y)

    # Write vocabulary
    with open('vocab.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([voc, voc_inv], f)

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int( dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Vocabulary Size: {:d}".format(len(voc)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    with tf.Session() as sess:

        seq_length = x_train.shape[1]
        num_class = y_train.shape[1]
        voc_size = len(voc)

        print('initialize cnn filter')
        print('sequence length %d,  number of class %d, vocab size %d' % (seq_length, num_class, voc_size))
        
        cnn = TextCNN(seq_length, num_class, voc_size, 64, [3,4,5], 64)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        def train_step(x_batch, y_batch):
            feed_dict = {
                cnn.input : x_batch,
                cnn.label : y_batch,
                cnn.dropout_keep_prob : 0.5
            }

            _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
            #print("TRAIN: step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))

        def evaluate(x_batch, y_batch):
            feed_dict = {
                cnn.input : x_batch,
                cnn.label : y_batch,
                cnn.dropout_keep_prob : 1.0
            }

            step, loss, accuracy = sess.run([global_step, cnn.loss, cnn.accuracy], feed_dict)
            print("TEST: step %d, loss %f, acc %f" % (step, loss, accuracy))

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # Generate batches
        batches = prep.batch_iter(
            list(zip(x_train, y_train)), batch_size, num_epochs)

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)

            current_step = tf.train.global_step(sess, global_step)

            if current_step % 100 == 0:
                print("\nEvaluation:")
                evaluate(x_dev, y_dev)
                print("")
            if current_step % 1000 == 0:
                 save_path = saver.save(sess, './textcnn.ckpt')
                 print('model saved : %s' % save_path)

if __name__ == '__main__':
    train()
