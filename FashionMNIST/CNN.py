#!/bin/python3

import numpy as np
import tensorflow as tf
import keras.backend as backend
from keras import layers, initializers, models

class CNN:
    @staticmethod
    def build(width, height):
        input_shape = (height, width, 1)
        inputs1 = layers.Input(shape = input_shape)
        conv1 = layers.Conv2D(filters = 256, kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu', name = 'conv1')(inputs1)
        conv2 = layers.Conv2D(filters = 256, kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu', name = 'conv2')(conv1)
        conv3 = layers.Conv2D(filters = 128, kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu', name = 'conv3')(conv2)
        flatten = layers.Flatten()(conv3)
        dense1 = layers.Dense(328, activation = 'relu', name = 'dense1')(flatten)
        dense2 = layers.Dense(192, activation = 'relu', name = 'dense2')(dense1)
        dropout = layers.Dropout(0.1)(dense2)
        dense3 = layers.Dense(10, activation = 'sigmoid', name = 'dense3')(dropout)
        model = models.Model(inputs = inputs1, outputs = dense3, name = "cnn")
        return model


train_set_size = 60000
test_set_size = 10000


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_pictures = np.reshape(train_images, [train_set_size, 28, 28, 1])

train_pictures_flat = np.reshape(train_images, [train_set_size, 784])

train_labels = np.reshape(train_labels, [train_set_size, ])

train_labels = np.eye(10)[train_labels]

test_pictures = np.reshape(test_images, [test_set_size, 28, 28, 1])

test_pictures_flat = np.reshape(test_images, [test_set_size, 784])

test_labels = np.reshape(test_labels, [test_set_size, ])

test_labels = np.eye(10)[test_labels]



model = CNN.build(28, 28)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', loss_weights = [1., 0.005], metrics = {'dense3' : 'accuracy'})

model.summary()


x = [train_pictures, train_labels]
y = [train_labels, train_pictures_flat]

val_x = [test_pictures, test_labels]
val_y = [test_labels, test_pictures_flat]

model.fit(train_pictures, train_labels, epochs = 15, batch_size = 128, validation_data = (test_pictures, test_labels))

res_x, res_y = model.evaluate(test_pictures, test_labels)

print(round(res_y, 4))
 
