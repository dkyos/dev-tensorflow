
# coding: utf-8

# \* *[Notice] I wrote thie code while following the examples in [Choi's Tesorflow-101 tutorial](https://github.com/sjchoi86/Tensorflow-101). And,  as I know, most of Choi's examples originally come from [Aymeric Damien's](https://github.com/aymericdamien/TensorFlow-Examples/) and  [Nathan Lintz's ](https://github.com/nlintz/TensorFlow-Tutorials) tutorials.*

# ## 4. Convolutional Neural Network with MNIST data

# In[1]:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# %matplotlib inline  


# ## Load MNIST data

# In[2]:

mnist      = input_data.read_data_sets('data', one_hot=True)
X_train   = mnist.train.images
Y_train = mnist.train.labels
X_test    = mnist.test.images
Y_test  = mnist.test.labels


# In[3]:

dimX = X_train.shape[1]
dimY = Y_train.shape[1]
nTrain = X_train.shape[0]
nTest = X_test.shape[0]
print ("Shape of (X_train, X_test, Y_train, Y_test)")
print (X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# ## Set parameters for my CNN structure

# In[4]:

pp = {
    'nLayerIn': dimX,
    'nLayerOut':  dimY,
    'sigma_init': 0.1, 
    'myDropProb': 0.7, 

    'nWin_conv1': 3,
    'nStr_conv1': 1,
    'nPad_conv1': 'SAME', # or 'VALID'
    'nWin_pool1': 2,
    'nStr_pool1': 2,
    'nPad_pool1': 'SAME',
    'nFeat1': 64,

    'nWin_conv2': 3,
    'nStr_conv2': 1,
    'nPad_conv2': 'SAME',
    'nWin_pool2': 2,
    'nStr_pool2': 2,
    'nPad_pool2': 'SAME',
    'nFeat2': 128,

    'dimX_mat': 28,   # 28*28 = 784
    'nDimReduce': 7,  # dimX_mat/nWin_pool1/nWin_pool2
    'nFull': 1024
}


# ## Build my CNN model

# In[5]:

W = {
    'W_conv1': tf.Variable(tf.truncated_normal([pp['nWin_conv1'], pp['nWin_conv1'], 1, pp['nFeat1']], stddev=pp['sigma_init'])),
    'W_conv2': tf.Variable(tf.truncated_normal([pp['nWin_conv2'], pp['nWin_conv2'], pp['nFeat1'], pp['nFeat2']], stddev=pp['sigma_init'])),
    'W_full': tf.Variable(tf.truncated_normal([pp['nDimReduce']*pp['nDimReduce']*pp['nFeat2'], pp['nFull']], stddev=pp['sigma_init'])),
    'W_out': tf.Variable(tf.truncated_normal([pp['nFull'], pp['nLayerOut']], stddev=pp['sigma_init']))
    
}
b = {
    'b_conv1': tf.Variable(tf.truncated_normal([pp['nFeat1']], stddev=pp['sigma_init'])),
    'b_conv2': tf.Variable(tf.truncated_normal([pp['nFeat2']], stddev=pp['sigma_init'])),
    'b_full': tf.Variable(tf.truncated_normal([pp['nFull']], stddev=pp['sigma_init'])),
    'b_out': tf.Variable(tf.truncated_normal([pp['nLayerOut']], stddev=pp['sigma_init']))
}


# In[6]:

def model_myCNN(_X, _W, _B, _dropout_prob, _pp):
       
    _X_mat = tf.reshape(_X, shape=[-1, _pp['dimX_mat'], _pp['dimX_mat'], 1])

    # L1: Convolution
    _L1_conv = tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(_X_mat, _W['W_conv1'], strides=[1, _pp['nStr_conv1'], _pp['nStr_conv1'], 1], padding=_pp['nPad_conv1'])
            , _B['b_conv1']))
    _L1_pool = tf.nn.max_pool(_L1_conv, ksize=[1, _pp['nWin_pool1'], _pp['nWin_pool1'], 1], strides=[1, _pp['nStr_pool1'], _pp['nStr_pool1'], 1], padding=_pp['nPad_pool1'])
    _L1_pool2 = tf.nn.dropout(_L1_pool, _dropout_prob)
    
    # L2: Convolution
    _L2_conv = tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(_L1_pool2, _W['W_conv2'], strides=[1, _pp['nStr_conv2'], _pp['nStr_conv2'], 1], padding=_pp['nPad_conv1'])
            , _B['b_conv2']))
    _L2_pool = tf.nn.max_pool(_L2_conv, ksize=[1, _pp['nWin_pool2'], _pp['nWin_pool2'], 1], strides=[1, _pp['nStr_pool2'], _pp['nStr_pool2'], 1], padding=_pp['nPad_pool1'])
    _L2_pool2 = tf.nn.dropout(_L2_pool, _dropout_prob)
    

    # L_full: Fully-connected
    _L2_pool2_vec = tf.reshape(_L2_pool2, [-1, _W['W_full'].get_shape().as_list()[0]])
    _L_full = tf.nn.relu(tf.add(tf.matmul(_L2_pool2_vec, _W['W_full']), _B['b_full']))
    _L_full2 = tf.nn.dropout(_L_full, _dropout_prob)
    
    # L_full: Output
    _L_out = tf.add(tf.matmul(_L_full2, _W['W_out']), _B['b_out'])
    
    # Return 
    out = {
        'X_mat': _X_mat,
        'L1_conv': _L1_conv, 'L1_pool': _L1_pool, 'L1_pool2': _L1_pool2, # After dropout
        'L2_conv': _L2_conv, 'L2_pool': _L2_pool, 'L2_pool2': _L2_pool2,
        'L_full': _L_full, 'L_full2': _L_full2, 'L_out': _L_out
    }
    return out


# ## Define variables and optimizer

# In[7]:

X = tf.placeholder(tf.float32, [None, dimX], name="input")
Y= tf.placeholder(tf.float32, [None, dimY], name="output")
dropout_prob = tf.placeholder(tf.float32, name="dropout")


# In[8]:

Y_pred_all = model_myCNN(X, W, b, dropout_prob, pp)
Y_pred = Y_pred_all['L_out']


# In[9]:

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_pred))
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
training_epochs = 5
display_epoch = 1
batch_size = 100   # For each time, we will use 100 samples to update ppeters 


# In[10]:

correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# ### Run the session

# In[11]:

# Because of the memory allocation problem in evaluation
divide_train = 50;
divide_test = 10;
nTrainSub = (int)(nTrain/divide_train);
nTestSub = (int)(nTest/divide_test);


# In[12]:

#with tf.Session() as sess:   
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for epoch in range(training_epochs):
    nBatch  = int(nTrain/batch_size)
    #myIdx =  np.random.permutation(nTrain)
    for ii in range(nBatch):
        X_Batch, Y_Batch = mnist.train.next_batch(batch_size)
        #X_Batch = X_train[myIdx[ii*batch_size:(ii+1)*batch_size],:]
        #Y_Batch = Y_train[myIdx[ii*batch_size:(ii+1)*batch_size],:]
        sess.run(optimizer, feed_dict={X:X_Batch, Y:Y_Batch, dropout_prob:pp['myDropProb']})

    if (epoch+1) % display_epoch == 0:
        # Because of the memory allocation problem in evaluation
        loss_temp = accuracy_train_temp = accuracy_test_temp = 0
        for jj in range(divide_train):
            myIdx1 = jj*nTrainSub
            myIdx2 = (jj+1)*nTrainSub
            loss_temp += sess.run(loss, feed_dict={X: X_train[myIdx1:myIdx2,:], Y:Y_train[myIdx1:myIdx2,:], dropout_prob:1.})
            accuracy_train_temp += accuracy.eval({X: X_train[myIdx1:myIdx2,:], Y:Y_train[myIdx1:myIdx2,:], dropout_prob:1.})
        for kk in range(divide_test):
            myIdx1 = kk*nTestSub
            myIdx2 = (kk+1)*nTestSub
            accuracy_test_temp += accuracy.eval({X: X_test[myIdx1:myIdx2,:], Y: Y_test[myIdx1:myIdx2,:], dropout_prob:1.}) 

        print ("(epoch {})".format(epoch+1))
        print ("[Loss / Tranining Accuracy / Test Accuracy] {:05.4f} / {:05.4f} / {:05.4f}".format(loss_temp/divide_train, accuracy_train_temp/divide_train, accuracy_test_temp/divide_test))
        print (" ")

print ("[Test Accuracy] {:05.4f}".format(accuracy_test_temp/divide_test))


# ## Let's see the learned features

# In[13]:

nExample = 223

Y_pred_all = model_myCNN(X, W, b, dropout_prob, pp)
X_mat = sess.run(Y_pred_all['X_mat'], feed_dict={X: X_train[nExample-1:nExample, :]})
L1_conv   = sess.run(Y_pred_all['L1_conv'], feed_dict={X: X_train[nExample-1:nExample, :]})

#L1_pool   = sess.run(Y_pred_all['L1_pool'], feed_dict={X: X_train[nExample-1:nExample, :]})
#L1_pool2   = sess.run(Y_pred_all['L1_pool2'], feed_dict={X: X_train[nExample-1:nExample, :]})
#L2_conv   = sess.run(Y_pred_all['L2_conv'], feed_dict={X: X_train[nExample-1:nExample, :]})
#L2_pool    = sess.run(Y_pred_all['L2_pool'], feed_dict={X: X_train[nExample-1:nExample, :]})
#L2_pool2    = sess.run(Y_pred_all['L2_pool2'], feed_dict={X: X_train[nExample-1:nExample, :]})
#L_full   = sess.run(Y_pred_all['L_full'], feed_dict={X: X_train[nExample-1:nExample, :]})
#L_full2     = sess.run(Y_pred_all['L_full2'], feed_dict={X: X_train[nExample-1:nExample, :]})


# In[ ]:

# (nExample)th Input
plt.matshow(X_mat[0, :, :, 0], cmap=plt.get_cmap('gray'))
plt.colorbar()
plt.show()


# In[ ]:

# Features
nFeature = 40
plt.matshow(L1_conv[0, :, :, nFeature], cmap=plt.get_cmap('gray'))
plt.colorbar()
plt.show() 


# In[ ]:

sess.close()

