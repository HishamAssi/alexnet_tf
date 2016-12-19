import tensorflow as tf
import numpy as np
import os

def create_normal_variable(shape):
    return tf.Variable(tf.random_normal(shape))

def conv2d(name, input, weight, bias, stride=1, padding="VALID"):
    x_input = input.get_shape()[-1]
    conv = tf.nn.conv2d(input, weight, [1, stride, stride, 1], padding=padding)
    return tf.nn.relu(tf.nn.bias_add(conv, bias), name)

def max_pool(name, input, size, stride):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides = [1, s, s, 1], padding = "VALID", name=name)

def norm(name, input, size=4):
    return tf.nn.lrn(input, size, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def create_net(input, dropout, num_classes, im_size):
    weights = {
        "wc1": create_normal_variable([11, 11, 3, 96])
        "wc2": create_normal_variable([5, 5, 96, 256])
        "wc3": create_normal_variable([3, 3, 256, 384])
        "wc4": create_normal_variable([3, 3, 384, 384])
        "wc5": create_normal_variable([3, 3, 384, 256])
        "wf1": create_normal_variable([4096, 4096])
        "wf2": create_normal_variable([4096, 4096])
        "out": create_normal_variable([4096, num_classes])
    }

    bias = {
        "bc1": create_normal_variable([96])
        "bc2": create_normal_variable([256])
        "bc3": create_normal_variable([384])
        "bc4": create_normal_variable([384])
        "bc5": create_normal_variable([256])
        "bf1": create_normal_variable([4096])
        "bf2": create_normal_variable([4096])
        "out": create_normal_variable([num_classes])
    }

    cl1 = conv2d("conv1", input, weights["wc1"], bias["bc1"], stride=4, padding="SAME")
    pool1 = max_pool("pool1", cl1, 3, 2)
    norm1 = norm("norm1", pool1)
    cl2 = conv2d("conv2", norm1, weights["wc2"], bias["bc2"])
    pool2 = max_pool("pool2", cl2, 3, 2)
    norm2 = norm("norm2", pool2)
    cl3 = conv2d("conv3", norm2, weights["wc3"], bias["bc3"])
    cl4 = conv2d("conv4", cl3, weights["wc4"], bias["bc4"])
    cl5 = conv2d("conv5", cl4, weights["wc5"], bias["bc5"])
    pool3 = max_pool("pool3", cl5, 3, 2)
    norm3 = norm("norm3", pool3)
    norm3 = tf.nn.dropout(norm3, dropout)
    fc1 = tf.reshape(norm3, [-1, weights["wf1"].get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.add(tf.matmul(dense1, weights["wf1"]), bias["bf1"]), name="fc1")
    drop1 = tf.nn.dropout(fc1, dropout)
    fc2 = tf.nn.relu(tf.add(tf.matmul(drop1, weights["wf2"]), bias["bf2"]), name="fc2")
    drop2 = tf.nn.dropout(fc2, dropout)
    out = tf.add(tf.matmul(drop2, weights["out"]), bias["out"])

    return out
