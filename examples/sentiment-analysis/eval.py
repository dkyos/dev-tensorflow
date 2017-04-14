#! /usr/bin/env python
import pickle

import tensorflow as tf
import numpy as np
import os
import time
import datetime
# import data_helpers
import prepare_konlpy as prep
from konlpy.tag import Twitter
from cnn import TextCNN
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

TEST_FILENAME = 'ratings_test.txt'
TEST_PICKLE = 'test.pickle'
VOCAB_PICKLE = 'vocab.pickle'

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1491539691/checkpoints", "Checkpoint directory from training run") 
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    if os.path.exists(TEST_PICKLE):
        with open(TEST_PICKLE, 'rb') as f:  # Python 3: open(..., 'rb')
            test_docs, y_test = pickle.load(f)
    else:
        # Load data
        print("Loading data...")

        test_data = prep.read_data(TEST_FILENAME)
    
        pos_tagger = Twitter()
    
        start_time = time.time()
        test_docs = [prep.tokenize(pos_tagger, row[1]) for row in test_data]
        y_test = [float(row[2]) for row in test_data]
        print('---- %s seconds elapsed ----' % (time.time() - start_time))
    
        # Saving the objects:
        with open(TEST_PICKLE, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([test_docs, y_test], f)

    y_test = prep.labeller(y_test)
    y_test = np.argmax(y_test, axis=1)

    x_raw = prep.pad_sentences(test_docs)

else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
with open(VOCAB_PICKLE, 'rb') as f:  # Python 3: open(..., 'rb')
    voc, voc_inv = pickle.load(f)
    # Build vocabulary
    x_test = prep.build_test_data(x_raw, voc)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input = graph.get_operation_by_name("input").outputs[0]
        # label = graph.get_operation_by_name("label").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = prep.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum((all_predictions == y_test)))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w', encoding='utf-8') as f:
    csv.writer(f).writerows(predictions_human_readable)

