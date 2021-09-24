#!/bin/python3

import numpy as np
import tensorflow as tf
import keras.backend as backend
from keras import layers, initializers, models
from PIL import Image

#a custom activation function (non-linear) from the paper to make sure all values represent probabilites in range [0, 1]
def squash(x, axis=-1):
    squared_norm = backend.sum(backend.square(x), axis, keepdims=True)
    safe_norm = backend.sqrt(squared_norm + backend.epsilon())
    squash_factor = squared_norm / (1 + squared_norm)
    unit_vector = x / safe_norm
    return squash_factor*unit_vector

def vec_length(vector, axis = -1):
    return backend.sqrt(backend.sum(backend.square(vector), axis=axis))

class MaskLayer(layers.Layer):
    def call(self, inputs):
        real_inputs, mask = inputs
        return backend.batch_dot(real_inputs, mask, [1, 1])
    def compute_output_shape(self, input_shape):
        return (None, input_shape[1][1])

class LengthLayer(layers.Layer):
    def call(self, inputs):
        #calculate length at the last dim
        return vec_length(inputs)
    def compute_output_shape(self, input_shape):
        #last dimension will be flattened to length
        return input_shape[:-1]

#custom loss function from the paper
#dims - [None, caps_n]
def margin_loss(y, y_hat):
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5
    L = y * backend.square(backend.maximum(0., m_plus - y_hat)) + lambda_ * (1-y) * backend.square(backend.maximum(0., y_hat - m_minus))
    return backend.sum(L, 1)

def euclidean_distance_loss(y_true, y_pred):
    #sqrt causes NaN error
    return backend.sqrt(backend.sum(backend.square(y_true - y_pred), axis=-1))


        
class CapsLayer(layers.Layer):
    def __init__(self, caps_n, dims_n, rounds, **kwargs):
        super(CapsLayer, self).__init__(**kwargs)
        self.caps_n = caps_n
        self.dims_n = dims_n
        self.rounds = rounds
        self.w_initializer = initializers.get('glorot_uniform')
        self.cc_initializer = initializers.get('zeros')
    def build(self, input_shape):
        #[?, in_caps_n, in_dims]
        self.in_caps_n = input_shape[1]
        self.in_dims_n = input_shape[2]
        #transformation matrix - used to sum all children predictions into one prediction and then dynamically route coupling coefficients
        #meant to be trained by network
        self.W = self.add_weight(shape = [self.caps_n, self.in_caps_n, self.dims_n, self.in_dims_n], initializer = self.w_initializer, trainable = True)
        #for keras
        self.built = True
        
    def call(self, inputs, training=None):
        #inputs are of shape [?, in_caps_n, in_dims_n]
        #we need a copy of inputs for every capsule
        inputs_expand = tf.expand_dims(inputs, 1)
        inputs_tiled = tf.tile(inputs_expand, [1, self.caps_n, 1, 1])
        #ranks have to match
        inputs_tiled = tf.expand_dims(inputs_tiled, 4)
        u_hat = tf.map_fn(lambda x: tf.matmul(self.W, x), elems = inputs_tiled)
        #routing algorithm - esentially k-clustering
        self.cc = tf.zeros(shape = [tf.shape(u_hat)[0], self.caps_n, self.in_caps_n, 1, 1])
        for i in range(0, self.rounds):
            #apply softmax, same shape like cc
            new_c = tf.nn.softmax(self.cc, axis=1)
            #predictions from all capsules from previous layers to all capsules in current layer
            #[?, caps_n, in_caps_n, dim_n, 1]
            weighted_pred = tf.multiply(new_c, u_hat)
            #the prediction for each capsule is squashed sum of all smaller predictions
            sum_pred = tf.reduce_sum(weighted_pred, 2, keepdims = True)
            #[?, caps_n, 1, dim_n, 1]
            output = squash(sum_pred, axis = -2)
            #calculate agreement
            #we need a copy of final predictions for every input capsule
            output_tiled = tf.tile(output, [1, 1, self.in_caps_n, 1, 1])
            #calculate scalar product according to paper
            agreement = tf.matmul(u_hat, output_tiled, transpose_a=True)
            if i != self.rounds - 1:
                self.cc = tf.add(self.cc, agreement)
        #get rid of unwanted dimensions
        return backend.reshape(output, [-1, self.caps_n, self.dims_n])


class CapsNet:
    @staticmethod
    def build(width, height, classes, routing_count):
        input_shape = (height, width, 1)
        inputs1 = layers.Input(shape = input_shape)
        conv1 = layers.Conv2D(filters = 256, kernel_size = 9, strides = 1, padding = 'valid', activation = 'relu', name = 'conv1')(inputs1)
        conv2 = layers.Conv2D(filters = 256, kernel_size = 9, strides = 2, padding = 'valid', activation = 'relu')(conv1)
        flat = layers.Reshape(target_shape = [-1, 8])(conv2)
        caps1 = layers.Lambda(squash)(flat)
        caps2 = CapsLayer(caps_n = classes, dims_n = 16, rounds = routing_count, name = 'caps2')(caps1)
        pred = LengthLayer(name = 'pred')(caps2)
        
        inputs2 = layers.Input(shape = (classes, ), name = 'inputs2')
        mask = MaskLayer(name = 'mask')((caps2, inputs2))
        dense1 = layers.Dense(512, activation = 'relu', name = 'dense1')(mask)
        dense2 = layers.Dense(1024, activation = 'relu', name = 'dense2')(dense1)
        dense3 = layers.Dense(784, activation = 'sigmoid', name = 'dense3')(dense2)
        model = models.Model(inputs = [inputs1, inputs2], outputs = [pred, dense3], name = "capsnet")
        return model



model = CapsNet.build(28, 28, 10, 3)

model.compile(optimizer = 'adam', loss = [margin_loss, euclidean_distance_loss], loss_weights = [1., 0.005], metrics = {'pred' : 'accuracy'})

file_in = open("input_data")
file_in2 = open("test_data")
pictures = []
labels = []
pictures2 = []
labels2 = []

train_set_size = 60000
test_set_size = 1000

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

model.fit(x, y, epochs = 5, batch_size = 64)

model_start_to_mask = models.Model(inputs = model.input, outputs = model.get_layer('mask').output)

recon_input = layers.Input(shape = (16,))

recon_next = recon_input

recon_next = model.get_layer('dense1')(recon_next)

recon_next = model.get_layer('dense2')(recon_next)

recon_next = model.get_layer('dense3')(recon_next)

model_mask_to_recon = models.Model(inputs = recon_input, outputs = recon_next)


for i in range(0, test_set_size):
    mask = model_start_to_mask.predict([np.array([test_pictures[i]]), np.array([test_labels[i]])])
    vec_pics = [None] * 16
    for j in range(0, 16):
        mask_curr = np.copy(mask)
        mask_curr[0][j] = mask[0][j]*(1/10)
        vec_pics[j] = np.reshape(model_mask_to_recon.predict(mask_curr), [28, 28])
        for k in range(2, 11):
            mask_curr = np.copy(mask)
            mask_curr[0][j] = mask[0][j]*(k/10)
            recon = np.reshape(model_mask_to_recon.predict(mask_curr), [28, 28])
            vec_pics[j] = np.concatenate((vec_pics[j], recon), axis = 1)
    for j in range(1, 16):
        vec_pics[0] = np.concatenate((vec_pics[0], vec_pics[j]), axis = 0)
    
    data = vec_pics[0]
    data *= 256
    data = np.array(data, dtype = np.uint8)
    image = Image.fromarray(data, mode = 'L')
    image.save("vec_changes/{}.png".format(i))
    




 
