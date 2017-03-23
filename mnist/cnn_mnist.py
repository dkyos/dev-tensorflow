#-*- coding: utf-8 -*-
import argparse
import sys
import time
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def init_weight(size_array, name,stddev=0.01):
    return tf.Variable(tf.random_normal(size_array,stddev=stddev),name=name)

def model(X,w1,w2,w3,w4,w_h,hidden_rate,last_rate):
    # X : (?,28,28,1)
    c1 = tf.nn.relu(tf.nn.conv2d(X,w1,strides=[1,1,1,1],padding='SAME'))  ## c1 :(?,28,28,32) -> 28x28 32 Ouput
    l1 = tf.nn.max_pool(c1, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') ## l1 : (?,14,14,32) -> 14x14 32 Ouput
    l1 = tf.nn.dropout(l1,hidden_rate)

    c2 = tf.nn.relu(tf.nn.conv2d(l1,w2,strides=[1,1,1,1],padding='SAME'))  ## c1 :(?,14,14,64) -> 14x14 64 ouput
    l2 = tf.nn.max_pool(c2, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') ## l1 : (?,7,7,64) ->7x7 64 Ouput
    l2 = tf.nn.dropout(l2,hidden_rate)

    c3 = tf.nn.relu(tf.nn.conv2d(l2,w3,strides=[1,1,1,1],padding='SAME'))  ## c1 :(?,7,7,128) -> 14x14 128 ouput
    l3 = tf.nn.max_pool(c3, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') ## l1 : (?,4,4,128) ->7x7 128 Ouput
    l3 = tf.reshape(l3,[-1,w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3,hidden_rate)

    l4 = tf.nn.relu(tf.matmul(l3,w4))
    l4 = tf.nn.dropout(l4,last_rate)

    hyp = tf.matmul(l4,w_h)
    return hyp

def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    X = tf.placeholder("float", [None, 28, 28, 1])
    Y = tf.placeholder("float", [None, 10])
    droprate_hidden  = tf.placeholder("float")
    droprate_last  = tf.placeholder("float")

    x_data = mnist.train.images
    y_data = mnist.train.labels
    test_x = mnist.test.images
    test_y = mnist.test.labels

    x_data = x_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img
    test_x = test_x.reshape(-1, 28, 28, 1)  # 28x28x1 input img

    w1 = init_weight([28,28,1,32],"w1") # 3x3x1 Input, 32 Ouput
    w2 = init_weight([14,14,32,64],"w2") # 3x3x32 Input, 64 Ouput
    w3 = init_weight([7,7,64,128],"w3") # 3x3x64 Input, 128 Ouput
    w4 = init_weight([4*4*128,625],"w4")
    w_h = init_weight([625,10],"wh")

    batch_size = 100

    hyp = model(X,w1,w2,w3,w4,w_h,droprate_hidden,droprate_last)
    print (hyp)
    print (Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=hyp))

    tf.train.AdamOptimizer
    optimizer =tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)
    predict_optimizer = tf.arg_max(hyp,1)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for _ in range(100):
        count_size = len(x_data)/batch_size
        count = 0
        for start,end in zip(range(0,len(x_data),batch_size),range(batch_size,len(y_data),batch_size)):
            count+=1
            if(count%10==0):
                test_index = np.arange(len(test_x)) #[1,2,3,..]
                np.random.shuffle(test_index)
                test_index = test_index[0:256]
                accruacy = np.mean(np.argmax(test_y[test_index],axis=1) \
                        == sess.run(predict_optimizer,feed_dict={X:test_x[test_index],Y:test_y[test_index],droprate_last:1.0,droprate_hidden:1.0}))
                print ("Accuracy :"+str(accruacy))

            sess.run(optimizer,\
                    feed_dict={X:x_data[start:end],Y:y_data[start:end],\
                    droprate_hidden:0.8,\
                    droprate_last:0.5})

            print (str(count)+"/"+str(count_size),start,end, \
                ("cost:"+str(\
                    sess.run(cost,\
                            feed_dict={X:x_data[start:end],Y:y_data[start:end],\
                            droprate_hidden:0.8,\
                            droprate_last:0.5}\
                        )\
                    )\
                ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
        type=str, 
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

