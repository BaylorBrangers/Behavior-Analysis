#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:36:09 2018

@author: baylor
"""


from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np
import tensorflow as tf
import accl_functions as af
import matplotlib.pyplot as plt
import os.path
#np.random.seed(23)
tf.reset_default_graph()

# Parameters
learning_rate = 0.0001
training_epochs = 100
batch_size = 32
display_step = 1

# Network Parameters
n_hidden_1 = 500 # 1st layer number of neurons
n_hidden_2 = 500 # 2nd layer number of neurons
n_input = 900 # MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)


# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
pred=tf.placeholder("float",)

initializer=tf.contrib.layers.xavier_initializer()

##XAVIER INITIALIZATION##
weights = {
    'h1': tf.Variable(initializer([n_input, n_hidden_1])),
    'h2': tf.Variable(initializer([n_hidden_1, n_hidden_2])),
    'weight_out': tf.Variable(initializer([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(initializer([n_hidden_1])),
    'b2': tf.Variable(initializer([n_hidden_2])),
    'bias_out': tf.Variable(initializer([n_classes]))
}

# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['weight_out']) + biases['bias_out']
    return out_layer

logits = multilayer_perceptron(X)
'''
# Define loss and optimizer
#loss_opo no weights
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
'''

###############################################################
#weighted_loss
loss_weights=tf.constant(([[0.2, 0.8]]))
loss_op=tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
        logits=logits,targets=Y,pos_weight=loss_weights))

################################################################

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


#######################################
## EVALUATE MODEL
# evaluates accuracy of softmax of logit output
#
pred=tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
#
## Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



############################################

#let's save some weights
saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

#load data from 
outfile='Train_Test_Data.npz'
npzfile = np.load('Train_Test_Data.npz')

training_data=npzfile['training_data']
training_labels=npzfile['training_labels']
test_set=npzfile['test_set']
test_labels=npzfile['test_labels']

#randomly shuffle all the data for training the network
shuffle_accl, shuffle_labels =af.shuffle_data(training_data, training_labels)

#set percentage of data to be used for validating
percentage=0.2

#Generate test set and training set
train_x,train_y,test_x,test_y=af.generate_test(shuffle_accl,shuffle_labels,percentage)


total_batch = int(train_y.shape[0]/batch_size)
avg_cost_array=np.zeros(training_epochs* total_batch)


soft_max_prediction=np.zeros((test_labels.shape[0],2),dtype='float32')

#used for plotting the difference between logits, alternative
#to using softmax
logit_diff=np.zeros((test_labels.shape[0],2),dtype='float32')

#file path for model saver
file_name='/home/baylor/Desktop/DATA/Behavior_Analysis/Behavior_Analysis/model.ckpt'

use_training= False

##Start traning ##
with tf.Session() as sess:

    sess.run(init)
    if use_training:
        #reloads previous training data
        saver.restore(sess, "/home/baylor/Desktop/DATA/Behavior_Analysis/Behavior_Analysis/model.ckpt")
    else:
        plt.figure() 
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            avg_cost_test = 0.
        
            for i in range(total_batch):
                
                #get batches for training
                batch_x,batch_y=af.batch_dat_data(train_x,train_y,i,batch_size)

                #calculate loss on test set, should be calculated before first training set

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
 
    
                # Compute average loss
                avg_cost += c / total_batch
 
                avg_cost_array[epoch*total_batch+i]= c    
    
            #Display logs per epoch step
            if epoch % display_step == 0:
                tc = sess.run([loss_op], feed_dict={X: test_x,
                                  Y: test_y})
    
                tc=float(tc[0])
                avg_cost_test = tc
    
    
                print("Epoch:", '%04d' % (epoch+1), "cost={:.15f}".format(avg_cost),
                      "cost={:.15f}".format(avg_cost_test))

                plt.plot((epoch+1),avg_cost,'.r')
                plt.plot((epoch+1),avg_cost_test,'xg')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.pause(1)
    
            if epoch%10 == 0:
                #save weights every 10 epochs
                save_path = saver.save(sess, "/home/baylor/Desktop/DATA/Behavior_Analysis/Behavior_Analysis/model" + str(epoch) + ".ckpt")
        
        print("Optimization Finished!")
 ##################################################################
#    plt.figure()
    
    for z in range(test_labels.shape[0]):
        soft_max_prediction[z,:]=pred.eval(feed_dict={X: test_set[[z],:]})
        logit_diff[z,:]=logits.eval(feed_dict={X: test_set[[z],:]})
    
    plt.figure()
    ax1=plt.subplot(211)
    #plt.subplot(3, 1, 2)
    plt.plot((logit_diff[:,0]-logit_diff[:,1]),'g')
    plt.xlabel('time')
    plt.ylabel('activation_model')


    ax2=plt.subplot(212,sharex=ax1)
    #plt.subplot(3, 1, 3)
    #plot mount behaior
    plt.plot(test_labels[:,0], 'r')
    #plot intromission data

    #plt.plot(a.intromission_lables_gt[:,0], 'r')
    plt.xlabel('time')
    plt.ylabel('activation_ground_truth')
    plt.show()

    print(" Test Accuracy:", accuracy.eval({X: test_set, Y: test_labels}))
#    print("Train Accuracy:", accuracy.eval({X: train_x, Y: train_y}))
#    print("Validation Accuracy:", accuracy.eval({X: test_x, Y: test_y}))
 ##################################################################



