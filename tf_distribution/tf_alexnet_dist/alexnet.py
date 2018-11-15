import tensorflow as tf
import numpy as np

tf_version = int(tf.__version__.split('.')[0])


def maxpooling2d(x, kHeight, kWidth, strideX, strideY, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, kHeight, kWidth, 1],
                          strides=[1, strideX, strideY, 1], padding=padding, name=name)


def dropout(x, keepprob, name=None):
    return tf.nn.dropout(x, keepprob, name)


def lrn(x, R, alpha, beta, name=None, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=R,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias,
                                              name=name)


def fc(x, inputdim, outputdim, reluFlag, name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[inputdim, outputdim], dtype='float',
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable('bias', [outputdim], dtype='float', initializer=tf.constant_initializer(0.0))

        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        return out


def conv(x, kHeight, kWidth, strideX, strideY, featureNum, name, padding='SAME', groups=1):
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("weights", shape=[kHeight, kWidth, channel, featureNum], dtype='float',
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable("bias", shape=[featureNum], dtype='float', initializer=tf.constant_initializer(0.0))
        out = tf.nn.conv2d(x, w, strides=[1, strideX, strideY, 1], padding=padding)
        return tf.nn.relu(tf.nn.bias_add(out, b))


class AlexNet(object):

    def __init__(self,
                 x,
                 keep_prob,
                 num_classes):
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob

        self.create()

    def create(self):
        conv1 = conv(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")
        lrn1 = lrn(conv1, 2, 2e-05, 0.75, "norm1")
        pool1 = maxpooling2d(lrn1, 3, 3, 2, 2, "pool1", "VALID")

        conv2 = conv(pool1, 5, 5, 1, 1, 256, "conv2", groups=1)
        lrn2 = lrn(conv2, 2, 2e-05, 0.75, "lrn2")
        pool2 = maxpooling2d(lrn2, 3, 3, 2, 2, "pool2", "VALID")

        conv3 = conv(pool2, 3, 3, 1, 1, 384, "conv3")

        conv4 = conv(conv3, 3, 3, 1, 1, 384, "conv4", groups=1)

        conv5 = conv(conv4, 3, 3, 1, 1, 256, "conv5", groups=1)
        pool5 = maxpooling2d(conv5, 3, 3, 2, 2, "pool5", "VALID")

        flatten1 = tf.reshape(pool5, [-1, 256 * 6 * 6])
        fc1 = fc(flatten1, 256 * 6 * 6, 4096, True, "fc6")
        dropout1 = dropout(fc1, self.KEEP_PROB)

        fc2 = fc(dropout1, 4096, 4096, True, "fc7")
        dropout2 = dropout(fc2, self.KEEP_PROB)

        self.fc3 = fc(dropout2, 4096, self.NUM_CLASSES, False, "fc8")
