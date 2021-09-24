#!/bin/python3

import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator


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


datagen = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.25, height_shift_range = 0.25, shear_range = 45, zoom_range=[0.5,1.5],)

datagen.fit(train_pictures)

i = 0

for x, y in datagen.flow(train_pictures, train_labels, batch_size = 1):
    data = np.reshape(x, [28, 28])
    data *= 256
    data = data.astype(np.uint8)
    arr = np.array([int(round(a, 4)) for a in np.reshape(x, [784])])
    print("{},{}".format(np.argmax(y), ",".join(str(a) for a in arr)))
    if i > 300000:
        break
    i += 1


