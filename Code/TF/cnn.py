import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Optimisation parameterss
learning_rate = 0.0001
epochs = 20
batch_size = 50

x = tf.placeholder(tf.float32, [None, 784]) #input vector
x_shaped = tf.reshape(x, [-1, 28, 28, 1])   #dynamic reshaping
y = tf.placeholder(tf.float32, [None, 10])  #output vector


def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    out_layer += bias

    out_layer = tf.nn.relu(out_layer)

    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, 
                               padding='SAME')

    return out_layer