from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

# Configuration
num_classes = 10
learning_rate = 0.001
training_steps = 200
batch_size = 128
display_step = 10

# Hyperparameters 
conv1_filters = 32
conv2_filters = 64
fc1_units = 1024

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train / 255., x_test / 255.

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

class ConvNet(Model):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = layers.Conv2D(conv1_filters, kernel_size = 5, activation = tf.nn.relu)
        self.maxpool1 = layers.MaxPool2D(2, strides = 2)
        self.conv2 = layers.Conv2D(conv2_filters, kernel_size = 3, activation = tf.nn.relu)
        self.maxpool2 = layers.MaxPool2D(2, strides = 2)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(fc1_units)
        self.dropout = layers.Dropout(rate = 0.5)
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training = False):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training = is_training)
        x = self.out(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return(x)

conv_net = ConvNet()

def cross_entropy_loss(x, y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = x)
    return tf.reduce_mean(loss)

def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis = -1)

optimizer = tf.optimizers.Adam(learning_rate)

def run_optimization(x, y):
    with tf.GradientTape() as g:
        pred = conv_net(x, is_training = True)
        loss = cross_entropy_loss(pred, y)
    
    trainable_variables = conv_net.trainable_variables
    gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimization(batch_x, batch_y)
    if step % display_step == 0:
        pred = conv_net(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("Step: %i, Loss: %f, Accuracy: %f" % (step, loss, acc))

pred = conv_net(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

import matplotlib.pyplot as plt
n_images = 10
test_images = x_test[:n_images]
predictions = conv_net(test_images)

for idx in range(n_images):
    plt.imshow(np.reshape(test_images[idx], [28, 28]), cmap = "gray")
    print("Model Prediction: %i" % np.argmax(predictions.numpy()[idx]))
    plt.show()