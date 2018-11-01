'''
    author: qwhu
    date: 2018.11.1
'''


import tensorflow as tf


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def inference(images):
    '''
    前向推断
    param：
          images: tensor，shape [None, 28, 28, 1]
    return
          softmax: tensor
    '''
    images = tf.cast(images, tf.float32)
    with tf.variable_scope('conv1'):
        weights = tf.get_variable('weights', [5, 5, 1, 32],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.relu(conv2d(images, weights) + biases)
        pool1 = max_pool_2x2(conv1)

    with tf.variable_scope('conv2'):
        weights = tf.get_variable('weights', [5, 5, 32, 64],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.relu(conv2d(pool1, weights) + biases)
        pool2 = max_pool_2x2(conv2)

    with tf.variable_scope('flatten'):
        flatten = tf.reshape(pool2, [-1, 7*7*64])

    with tf.variable_scope('fc1'):
        weights = tf.get_variable('weights', [7*7*64, 1024],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [1024],initializer=tf.constant_initializer(0.0))

        fc1 = tf.nn.relu(tf.matmul(flatten, weights) + biases)

    with tf.variable_scope('fc2'):
        weights = tf.get_variable('weights', [1024, 10],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [10],initializer=tf.constant_initializer(0.0))

        softmax = tf.matmul(fc1, weights) + biases

    return softmax


def losses(out, labels):
    '''
    计算loss
    param
          out: tensor，此处使用的softmax_cross_entropy_with_logits的输入为未经过softmax的tensor
          labels:  labels one-hot vector
                   sparse_softmax_cross_entropy_with_logits的label为class index
    return:
          loss
    '''
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=out, name='cross_entropy')
    loss = tf.reduce_mean(cross_entropy, name='losses')
    return loss


def accuray(out, labels):
    labels = tf.cast(labels, tf.int64)
    correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(labels, 1))
    accuray = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuray


def train(loss):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    return train_op