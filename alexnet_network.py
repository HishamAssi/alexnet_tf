import tensorflow as tf
import numpy as np
import os
from utils import Dataset
import params as pm

def create_normal_variable(shape):
    return tf.Variable(tf.random_normal(shape))

def conv2d(name, input, weight, bias, stride=1, padding="VALID"):
    x_input = input.get_shape()[-1]
    conv = tf.nn.conv2d(input, weight, [1, stride, stride, 1], padding=padding)
    return tf.nn.relu(tf.nn.bias_add(conv, bias), name)

def max_pool(name, input, size, stride):
    return tf.nn.max_pool(input, ksize=[1, size, size, 1], strides = [1, stride, stride, 1], padding = "VALID", name=name)

def norm(name, input, size=4):
    return tf.nn.lrn(input, size, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def create_net(input, dropout, num_classes, im_size):
    weights = {
        "wc1": create_normal_variable([11, 11, 3, 96]),
        "wc2": create_normal_variable([5, 5, 96, 256]),
        "wc3": create_normal_variable([3, 3, 256, 384]),
        "wc4": create_normal_variable([3, 3, 384, 384]),
        "wc5": create_normal_variable([3, 3, 384, 256]),
        "wf1": create_normal_variable([1024, 4096]),
        "wf2": create_normal_variable([4096, 4096]),
        "out": create_normal_variable([4096, num_classes])
    }

    bias = {
        "bc1": create_normal_variable([96]),
        "bc2": create_normal_variable([256]),
        "bc3": create_normal_variable([384]),
        "bc4": create_normal_variable([384]),
        "bc5": create_normal_variable([256]),
        "bf1": create_normal_variable([4096]),
        "bf2": create_normal_variable([4096]),
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
    fc1 = tf.reshape(norm3, [-1, int(np.prod(pool3.get_shape()[1:]))])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, weights["wf1"]), bias["bf1"]), name="fc1")
    drop1 = tf.nn.dropout(fc1, dropout)
    fc2 = tf.nn.relu(tf.add(tf.matmul(drop1, weights["wf2"]), bias["bf2"]), name="fc2")
    drop2 = tf.nn.dropout(fc2, dropout)
    out = tf.add(tf.matmul(drop2, weights["out"]), bias["out"])

    return out

def main():
    training = Dataset(pm.TRAIN_PATH, ".jpg")
    testing = Dataset(pm.TEST_PATH, ".jpg")

    lr = pm.LEARN_RATE

    decay = pm.DECAY_RATE

    batch_size = pm.BATCH_SIZE

    display_step = pm.DISPLAY_STEP

    classes_num = training.num_labels

    dropout = pm.DROPOUT

    img_size = pm.IMAGE_SIZE
    img_channels = pm.IMAGE_CHANNELS

    x = tf.placeholder(tf.float32, [None, img_size, img_size, img_channels])
    y = tf.placeholder(tf.float32, [None, classes_num])

    keep_prob = tf.placeholder(tf.float32)

    pred = create_net(x, keep_prob, classes_num, img_size)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

    step = tf.Variable(0, trainable=False)
    lr_with_decay = tf.train.exponential_decay(lr, step, 1000, decay, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_with_decay).minimize(cost, global_step=step)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("keep_prob", keep_prob)
    tf.add_to_collection("pred", pred)
    tf.add_to_collection("accuracy", accuracy)

    with tf.Session() as sess:
        sess.run(init)
        step = 1
        while step < 3000:
            batch_ys, batch_xs = training.nextBatch(batch_size)

            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                rate = sess.run(lr_with_decay)
                print "learning_rate: " + str(rate) + "; Iter: " + str(step) + "; Loss: " + "{:.6f}".format(loss) + "; Training Accuracy: " + "{:.5f}".format(acc)

            if step % 100 == 0:
                saver.save(sess, "checkpoints/inference.ckpt", global_step=step*batch_size)

            step += 1

    print "Done."

    step_test = 1
    while step_test * batch_size < len(testing):
        testing_ys, testing_xs = testing.nextBatch(batch_size)
        print "Testing Accuracy: " + str(sess.run(accuracy, feed_dict={x: testing_xs, y: testing_ys, keep_prob: 1}))
        step_test += 1

if __name__ == "__main__":
    main()
