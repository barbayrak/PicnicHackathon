#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
import tensorflow as tf
from tensorflow.python.framework import ops
import scipy
from scipy import ndimage
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import cv2
import time
import gc    

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

#Tensor placeholder creation function
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, shape =[None,n_H0,n_W0,n_C0])
    Y = tf.placeholder(tf.float32, shape =[None,n_y])
    return X, Y


# In[9]:


def initialize_parameters():
    W1 = tf.get_variable("W1", [11,11,3,96], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [5, 5, 96, 256], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W3 = tf.get_variable("W3", [3,3,256,384], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W4 = tf.get_variable("W4", [3,3,384,384], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W5 = tf.get_variable("W5", [3,3,384,256], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    
    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4,
                  "W5": W5}
    return parameters


# In[10]:


#Forward propogation function implemented
#CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
def forward_propagation(X, parameters):

    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    W5 = parameters['W5']
    
    #1 Convolution -> RELU -> Max Pool

    Z1 = tf.nn.conv2d(X,W1, strides = [1,4,4,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1,ksize= [1,3,3,1],strides=[1,2,2,1],padding = "VALID")

    #2 Convolution -> RELU -> Max Pool
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2,ksize= [1,3,3,1],strides=[1,2,2,1],padding = "VALID")

    #3 Convolution -> RELU 
    Z3 = tf.nn.conv2d(P2,W3, strides = [1,1,1,1], padding = 'SAME')
    A3 = tf.nn.relu(Z3)
    #P3 = tf.nn.max_pool(A3,ksize= [1,3,3,1],strides=[1,2,2,1],padding = "VALID")

    #4 Convolution -> RELU 
    Z4 = tf.nn.conv2d(A3,W4, strides = [1,1,1,1], padding = 'SAME')
    A4 = tf.nn.relu(Z4)
    #P4 = tf.nn.max_pool(A4,ksize= [1,3,3,1],strides=[1,2,2,1],padding = "VALID")
    print("Fourth Convolution :",A4.shape)
    #5 Convolution -> RELU -> Max Pool
    Z5 = tf.nn.conv2d(A4,W5, strides = [1,1,1,1], padding = 'SAME')
    A5 = tf.nn.relu(Z5)
    P5 = tf.nn.max_pool(A5,ksize= [1,3,3,1],strides=[1,2,2,1],padding = "VALID")

    P5 = tf.contrib.layers.flatten(P5)

    print(P5.shape)
    F1 = tf.contrib.layers.fully_connected(P5, 4096 ,activation_fn=tf.nn.relu)
    print(F1.shape)
    F2 = tf.contrib.layers.fully_connected(F1, 4096 ,activation_fn=tf.nn.relu)
    print(F2.shape)
    F3 = tf.contrib.layers.fully_connected(F2, 25 ,activation_fn=None)
    print(F3.shape)
    return F3



# In[11]:


#Compute cost function with softmax entropy
def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z3, labels = Y))
    return cost


# In[12]:


#Model implementation CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
def model(X_train_m, Y_train_m, X_test_m, Y_test_m, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):

    print("Started")
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train_m.shape             
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    
    X, Y = create_placeholders(n_H0,n_W0,n_C0,n_y)

    
    # Initialize 
    parameters = initialize_parameters()
    print("Parameters Initialized")
    
    # Forward propagation
    Z3 = forward_propagation(X,parameters)
    print("Forward Propogated")

    #Compute cost
    cost = compute_cost(Z3,Y)
    print("Cost computed")
    
    # Backpropagation with AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    print("Optimized with AdamOptimizer")
    
    saver = tf.train.Saver()

    # Initialize all the variables globally
    init = tf.global_variables_initializer()
    print("Global variables initialized")
     
    # Start session to run the model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        
        sess.run(init)
        print("Tensorflow Session Started")
        
        # Training Loop
        for epoch in range(num_epochs):
            #print("Epoch Started : " , epoch)
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            
            #print("Random mini batches started ")
            minibatches = random_mini_batches(X_train_m, Y_train_m, minibatch_size, seed)

            #print("For minibatch started" )
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches
                
            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()


        print("Plot Showed")
        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        print("Correct Predictions Finished")
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)

        #Save Model
        save_path = saver.save(sess, "./cnn_model.ckpt")
        print("Model saved in path: %s" % save_path)

        print("Accuracy Written")

        #train_accuracy = accuracy.eval({X: X_train_m, Y: Y_train_m})
        #print("Train Accuracy:", train_accuracy)
        test_accuracy = accuracy.eval({X: X_test_m, Y: Y_test_m})
        print("Test Accuracy:", test_accuracy)
                
        #return train_accuracy, test_accuracy, parameters
        return test_accuracy, test_accuracy, parameters


# In[13]:


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# In[ ]:


X_augmented_images = np.load('X_resized_images_train.npy')
Y_augmented_labels = np.load('Y_resized_labels_train.npy')

print("Manupulated X(images) shape : ",X_augmented_images.shape)
print("Manupulated Y(label) shape: ",Y_augmented_labels.shape)

X_manupulated = X_augmented_images
Y_manupulated = Y_augmented_labels

#Encode labels into integers and one_hot_fix
Y_one_hot = np.unique(Y_manupulated, return_inverse=True)[1]
max_number_of_labels = (np.amax(Y_one_hot) + 1)
Y_one_hot_fixed = convert_to_one_hot(Y_one_hot,max_number_of_labels).T

print(X_manupulated[0])
# Train/Dev Split
X_manupulated = X_manupulated/255.
X_train,X_dev,Y_train,Y_dev = train_test_split(X_manupulated,Y_one_hot_fixed, test_size = 0.10,shuffle=True)
print("X_train shape : " , X_train.shape)
print("X_dev shape : " , X_dev.shape)
print("Y_train shape : " , Y_train.shape)
print("Y_dev shape : " , Y_dev.shape)

_, _, parameters = model(X_train, Y_train, X_dev, Y_dev,minibatch_size = 64,learning_rate = 0.01,num_epochs = 1000)




