
# coding: utf-8

# \* *[Notice] I wrote thie code while following the examples in [Choi's Tesorflow-101 tutorial](https://github.com/sjchoi86/Tensorflow-101). And,  as I know, most of Choi's examples originally come from [Aymeric Damien's](https://github.com/aymericdamien/TensorFlow-Examples/) and  [Nathan Lintz's ](https://github.com/nlintz/TensorFlow-Tutorials) tutorials.*

# ## 3. Multilayer Perceptron with MNIST data

# In[1]:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#%matplotlib inline  


# ### Load MNIST data

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


# ## Define my neural network structure

# In[4]:

nLayer0 = dimX
nLayer1 = 256
nLayer2 = 256
nLayer3 =  dimY
sigma_init = 0.1   # For randomized initialization


# In[7]:

W = {
    'W1': tf.Variable(tf.random_normal([nLayer0, nLayer1], stddev = sigma_init)),
    'W2': tf.Variable(tf.random_normal([nLayer1, nLayer2], stddev = sigma_init)),
    'W3': tf.Variable(tf.random_normal([nLayer2, nLayer3], stddev = sigma_init))
}
b = {
    'b1': tf.Variable(tf.random_normal([nLayer1])),
    'b2': tf.Variable(tf.random_normal([nLayer2])),
    'b3': tf.Variable(tf.random_normal([nLayer3]))
}


# In[8]:

def model_myNN(_X, _W, _b):
    Layer1 = tf.nn.sigmoid(tf.add(tf.matmul(_X,_W['W1']), _b['b1']))
    Layer2 = tf.nn.sigmoid(tf.add(tf.matmul(Layer1,_W['W2']), _b['b2']))
    Layer3 = tf.add(tf.matmul(Layer2,_W['W3']), _b['b3'])    
    #Layer3 = tf.nn.sigmoid(tf.add(tf.matmul(Layer2,_W['W3']), _b['b3']))
    return Layer3


# In[9]:

X = tf.placeholder(tf.float32, [None, dimX], name="input")
Y= tf.placeholder(tf.float32, [None, dimY], name="output")


# In[10]:

Y_pred = model_myNN(X, W, b)


# ### Define loss function, optimizer, measurer

# We use *softmax_cross_entropy()* and *AdamOptimizer()*

# In[9]:

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_pred))


# In[10]:

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
training_epochs = 30
display_epoch = 5
batch_size = 100   # For each time, we will use 100 samples to update parameters 


# In[11]:

correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# ### Run the session

# We use *with* for load a TF session

# In[12]:

with tf.Session() as sess:
    #sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        nBatch  = int(nTrain/batch_size)
        myIdx =  np.random.permutation(nTrain)
        for ii in range(nBatch):
            X_batch = X_train[myIdx[ii*batch_size:(ii+1)*batch_size],:]
            Y_batch = Y_train[myIdx[ii*batch_size:(ii+1)*batch_size],:]
            #print X_batch.shape, Y_batch.shape
            sess.run(optimizer, feed_dict={X:X_batch, Y:Y_batch})
          
        if (epoch+1) % display_epoch == 0:
            loss_temp = sess.run(loss, feed_dict={X: X_train, Y:Y_train}) 
            accuracy_temp = accuracy.eval({X: X_train, Y:Y_train})
            print ("(epoch {})".format(epoch+1) )
            print ("[Loss / Tranining Accuracy] {:05.4f} / {:05.4f}".format(loss_temp, accuracy_temp))
            print (" ")
            
    print ("[Test Accuracy] ",  accuracy.eval({X: X_test, Y: Y_test}))

