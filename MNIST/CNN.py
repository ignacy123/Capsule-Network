#!/bin/python3

import numpy as np
import tensorflow as tf
import keras.backend as backend
from keras import layers, initializers, models
from PIL import Image

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



model = CNN.build(28, 28)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', loss_weights = [1., 0.005], metrics = {'dense3' : 'accuracy'})

model.summary()

file_in = open("input_data")
file_in2 = open("test_data")
pictures = []
labels = []
pictures2 = []
labels2 = []

train_set_size = 60000
test_set_size = 10000

for i in range(0, train_set_size):
    line = file_in.readline()
    vector = [int(float(x)) for x in line.split(",")]
    labels.append(vector[0])
    vector = [x/256 for x in vector]
    vector = np.array(vector)
    pictures.append(vector[1:])
for i in range(0, test_set_size):
    line = file_in2.readline()
    vector = [int(float(x)) for x in line.split(",")]
    labels2.append(vector[0])
    vector = [x/256 for x in vector]
    vector = np.array(vector)
    pictures2.append(vector[1:])
train_pictures = np.array(pictures)
train_labels = np.array(labels)
train_labels = np.eye(10)[train_labels]
test_pictures = np.array(pictures2)
test_labels = np.array(labels2)
org_test_labels = test_labels
test_labels = np.eye(10)[test_labels]
train_pictures_flat = train_pictures
test_pictures_flat = test_pictures
train_pictures = np.reshape(train_pictures, (train_set_size, 28, 28, 1))
test_pictures = np.reshape(test_pictures, (test_set_size, 28, 28, 1))

file_in.close()
file_in2.close()

x = [train_pictures, train_labels]
y = [train_labels, train_pictures_flat]

val_x = [test_pictures, test_labels]
val_y = [test_labels, test_pictures_flat]

model.fit(train_pictures, train_labels, epochs = 100, batch_size = 128, validation_data = (test_pictures, test_labels))

res_x, res_y = model.evaluate(test_pictures, test_labels)

print(round(res_y, 4))
 
