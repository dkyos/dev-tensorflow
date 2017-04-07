#! /usr/bin/env python

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
from tensorflow.contrib import learn

################################################################
# Parameters

TRAIN_FILENAME = 'ratings_train.txt'
TRAIN_PICKLE = 'train.pickle'
VOCAB_PICKLE = 'vocab.pickle'

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")   # success : 64
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")   # success : 200
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

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
train_docs_p = prep.pad_sentences(train_docs)

# Build vocabulary
voc, voc_inv = prep.build_vocab(train_docs_p)

# Write vocabulary
with open(VOCAB_PICKLE, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([voc, voc_inv], f)

x, y = prep.build_input_data(train_docs_p, train_labels, voc)
#print(x)
#print(y)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int( FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(voc)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)

with tf.Session(config=session_conf) as sess:

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

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", cnn.loss)
    acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    def train_step(x_batch, y_batch):
        feed_dict = {
            cnn.input : x_batch,
            cnn.label : y_batch,
            cnn.dropout_keep_prob : 0.5
        }
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        #print("TRAIN: {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

    def evaluate(x_batch, y_batch, writer=None):
        feed_dict = {
            cnn.input : x_batch,
            cnn.label : y_batch,
            cnn.dropout_keep_prob : 1.0
        }

        step, summaries, loss, accuracy = sess.run(
            [global_step, dev_summary_op, cnn.loss, cnn.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("TEST: {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # Generate batches
    batches = prep.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)

        current_step = tf.train.global_step(sess, global_step)

        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            dev_batches = prep.batch_iter(list(zip(x_dev, y_dev)), 5000, 1)
            for dev_batch in dev_batches:
                x_dev_batch, y_dev_batch = zip(*dev_batch)
                evaluate(x_dev_batch, y_dev_batch, writer=dev_summary_writer)
            print("")
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))

