# coding: utf-8

# \* *[Notice] I wrote thie code while following the examples in [Choi's Tesorflow-101 tutorial](https://github.com/sjchoi86/Tensorflow-101). And,  as I know, most of Choi's examples originally come from [Aymeric Damien's](https://github.com/aymericdamien/TensorFlow-Examples/) and  [Nathan Lintz's ](https://github.com/nlintz/TensorFlow-Tutorials) tutorials.*

# ## 2. Logistic Regression with MNIST data

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


# ### Plot an example image of MNIST data

# In[4]:

myIdx = 36436   # any number
img   = np.reshape(X_train[myIdx, ], (28, 28)) # 28 * 28 = 784

plt.matshow(img, cmap=plt.get_cmap('gray'))
plt.show()


# ### Write a TF graph

# In[5]:

X = tf.placeholder(tf.float32, [None, dimX], name="input")
Y= tf.placeholder(tf.float32, [None, dimY], name="output")
W = tf.Variable(tf.zeros([dimX, dimY]), name="weight")
b = tf.Variable(tf.zeros([dimY]), name="bias")


# The output of the logic regression is  $softmax(Wx+b)$
# 
# Note that the dimension of *Y_pred* is *(nBatch, dimY)*

# In[6]:

Y_pred = tf.nn.softmax(tf.matmul(X, W) + b)


# We use a cross-entropy loss function,  $loss = -\Sigma y'\log(y)$
# 
# *reduce_sum(X, 1)* returns the sum across the columes of the tensor *X* 
# 
# *reduce_mean(X)* returns the mean value for all elements of the tensor *X*

# In[7]:

loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(Y_pred), reduction_indices=1))


# In[8]:

learning_rate = 0.005
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
training_epochs = 50
display_epoch = 5
batch_size = 100   # For each time, we will use 100 samples to update parameters 


# ### Compare prediction with the true value

# *argmax(X,1)*  returns the index of maximum value (which represents the label in this example) across the colums of the tensor *X*

# In[16]:

correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))    


# In[17]:

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# ### Run the session

# We use *with* for load a TF session

# In[18]:

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for epoch in range(training_epochs):
        nBatch  = int(nTrain/batch_size)
        myIdx =  np.random.permutation(nTrain)
        for ii in range(nBatch):
            X_batch = X_train[myIdx[ii*batch_size:(ii+1)*batch_size],:]
            Y_batch = Y_train[myIdx[ii*batch_size:(ii+1)*batch_size],:]
            sess.run(optimizer, feed_dict={X:X_batch, Y:Y_batch})
          
        if (epoch+1) % display_epoch == 0:
            loss_temp = sess.run(loss, feed_dict={X: X_train, Y:Y_train}) 
            accuracy_temp = accuracy.eval({X: X_train, Y:Y_train})
            print ("(epoch {})".format(epoch+1) )
            print ("[Loss / Training Accuracy] {:05.4f} / {:05.4f}".format(loss_temp,
                accuracy_temp))
            print (" ")
            
    print ("[Test Accuracy] ",  accuracy.eval({X: X_test, Y: Y_test})   )

