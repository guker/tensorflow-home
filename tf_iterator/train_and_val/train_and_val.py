'''

   1. 使用数据可初始化迭代器做input pipline
   2. 边训练边验证
      每个epoch起止可以通过捕捉OutOfRange异常来界定

'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dataset
import model

BATCHSIZE = 128
NCLASS = 10

# 返回训练集的数据迭代器
iterator = dataset.get_batches_v1('train_mnist.tfrecords', BATCHSIZE, NCLASS)
# 返回验证机的数据迭代器
val_iterator = dataset.get_batches_v1('val_mnist.tfrecords', BATCHSIZE, NCLASS, False)

# 调用get_next方法，得到batch tensor
images, labels = iterator.get_next()
val_images, val_labels = val_iterator.get_next()

'''
   这里使用feed_dict方式给数据进行训练
   shape=(None, 28, 28, 1), None表示在该维度可以接受任何大小的数值
   这样，可以适应在进行训练的时候，遍历每个epoch的时候，最后的数据不足构成一个指定的batch size的情形
   
'''
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.int32, shape=(None, NCLASS))

# 计算图构建，训练与验证共用同一个计算图
train_out = model.inference(x)
loss = model.losses(train_out, y)
train_acc = model.accuray(train_out, y)
train_step = model.train(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


num_epoches = 200

for epoch in range(num_epoches):
    # 初始化迭代器，因为迭代器没有设置repeat函数，所以遍历每个epoch的最后，都会抛出OutOfRange异常
    # 捕捉此异常，可以界定epoch的起止
    # 每次初始化迭代器，迭代器都会重新回到epoch的开头
    sess.run(iterator.initializer)
    while True:
        try:
            tra_batch, tra_lab = sess.run([images, labels])
            sess.run(train_step, feed_dict={
                x: tra_batch, y: tra_lab
            })
        except tf.errors.OutOfRangeError:
           break
    sess.run(iterator.initializer)
    total_correct = 0
    total = 0
    # 计算训练集的准确率
    while True:
        try:
            tra_batch, tra_lab = sess.run([images, labels])
            acc = sess.run(train_acc, feed_dict={
                x: tra_batch, y: tra_lab
            })
            total += tra_batch.shape[0]
            total_correct += tra_batch.shape[0]*acc

        except tf.errors.OutOfRangeError:
            print('epoch: %s, train acc: %s' % (epoch, total_correct / total))
            total = 0
            total_correct = 0
            break
    # 计算验证机的准确率
    sess.run(val_iterator.initializer)
    while True:
        try:
            val_batch, val_lab = sess.run([val_images, val_labels])
            acc = sess.run(train_acc, feed_dict={
                x: val_batch, y: val_lab
            })
            total += val_batch.shape[0]
            total_correct += val_batch.shape[0]*acc

        except tf.errors.OutOfRangeError:
            print('epoch: %s, val acc: %s' % (epoch, total_correct / total))
            break








