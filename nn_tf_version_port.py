#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ported from TensorFlow 1.x to TensorFlow 2.x
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os.path
# Import your accl_functions module
import accl_functions as af  

# Parameters
learning_rate = 0.0001
training_epochs = 300
batch_size = 32
display_step = 1

# Network Parameters
n_hidden_1 = 500  # 1st layer number of neurons
n_hidden_2 = 500  # 2nd layer number of neurons
n_input = 900     # Input dimension
n_classes = 2     # Number of classes

# Create the model using Keras Sequential API
def create_model():
    initializer = tf.keras.initializers.GlorotUniform()  # Xavier initialization
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(n_hidden_1, activation='relu', 
                             kernel_initializer=initializer, 
                             bias_initializer=initializer,
                             input_shape=(n_input,)),
        tf.keras.layers.Dense(n_hidden_2, activation='relu',
                             kernel_initializer=initializer,
                             bias_initializer=initializer),
        tf.keras.layers.Dense(n_classes, 
                             kernel_initializer=initializer,
                             bias_initializer=initializer)
    ])
    
    return model

# Create a custom loss function for weighted cross entropy
class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, pos_weight=tf.constant([[0.2, 0.8]]), **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight
        
    def call(self, y_true, y_pred):
        return tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                labels=y_true, 
                logits=y_pred, 
                pos_weight=self.pos_weight
            )
        )

# Create model
model = create_model()

# Compile the model with our custom loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=WeightedBinaryCrossentropy(),
    metrics=['accuracy']
)

# Load data
npzfile = np.load('Train_Test_Data.npz')
training_data = npzfile['training_data']
training_labels = npzfile['training_labels']
test_set = npzfile['test_set']
test_labels = npzfile['test_labels']

# For single session data
single_session = np.load('Train_Test_Data_Single_Video.npz')
test_set_single = single_session['test_set_single']
test_labels_single = single_session['test_labels_single']

# File path for model saver
checkpoint_path = '/home/baylor/Desktop/DATA/Behavior_Analysis/Model_Weights/model.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create callback for saving checkpoints
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_freq=10 * (len(training_data) // batch_size),
    verbose=1
)

use_training = True
single_trial = True

if not use_training:
    # Randomly shuffle all the data for training the network
    shuffle_accl, shuffle_labels = af.shuffle_data(training_data, training_labels)
    
    # Set percentage of data to be used for validating
    percentage = 0.2
    
    # Generate test set and training set
    train_x, train_y, validation_x, validation_y = af.generate_test(
        shuffle_accl, shuffle_labels, percentage
    )
    
    # Create a TensorBoard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )
    
    # Training history plot
    plt.figure()
    
    # Train the model
    history = model.fit(
        train_x, train_y,
        batch_size=batch_size,
        epochs=training_epochs,
        validation_data=(validation_x, validation_y),
        callbacks=[checkpoint_callback, tensorboard_callback],
        verbose=2
    )
    
    # Plot training history
    plt.plot(history.history['loss'], 'r')
    plt.plot(history.history['val_loss'], 'g')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(['Training', 'Validation'])
    plt.show()
    
    print("Optimization Finished!")

# For prediction and evaluation
soft_max_prediction = np.zeros((test_labels.shape[0], 2), dtype='float32')
soft_max_prediction_single = np.zeros((test_labels_single.shape[0], 2), dtype='float32')
logit_diff = np.zeros((test_labels.shape[0], 2), dtype='float32')
logit_diff_single = np.zeros((test_labels_single.shape[0], 2), dtype='float32')

# Load weights if use_training is True
if use_training:
    # Adjust this path to point to your saved model weights
    model.load_weights("/home/baylor/Desktop/DATA/Behavior_Analysis/Model_Weights/model250.ckpt")

# Make predictions on single session test data
logits_single = model.predict(test_set_single)
soft_max_prediction_single = tf.nn.softmax(logits_single).numpy()
logit_diff_single = logits_single

# Visualize the results
plt.figure()
ax1 = plt.subplot(211)
plt.plot((logit_diff_single[:, 0] - logit_diff_single[:, 1]), 'g')
plt.title('Logit Difference \n 300 Epochs')
plt.xlabel('time')
plt.ylabel('activation_model')

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(test_labels_single[:, 0], 'r')
plt.xlabel('time')
plt.ylabel('activation_ground_truth')
plt.show()

plt.figure()
a = logit_diff_single[:, 0] - logit_diff_single[:, 1]
# Don't care about values less than 0
a[a <= 0] = 0
a_norm = a / np.max(a)
plt.plot(a_norm)
plt.plot(test_labels_single[:, 0], 'r')
plt.title('Normalized Logit Difference')
plt.xlabel('time (ms)')
plt.ylabel('activation_model')
plt.show()

# Evaluate model on single test file
test_loss, test_acc = model.evaluate(test_set_single, test_labels_single, verbose=2)
print(f"Test Accuracy Single File: {test_acc:.4f}")
