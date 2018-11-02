'''

   1. 使用数据 feedable iterator做input pipline
   2. 因为feedable iterator 可以通过placeholder指定数据集，所以
      可以不使用feed_dict方式给batch数据，而是直接把input pipline
      接入graph

   3. 边训练边验证
      每个epoch起止可以通过捕捉OutOfRange异常来界定
   4. feedable iterator与可重新初始化迭代器功能类似，目前看有点复杂

    测试feedable iterator的功能
    经测试，得出如下结论：
    1. feedable iterator与可重新初始化迭代器的功能相同，但是迭代器之间的切换不需要从
       数据集的开头初始化迭代器，可通过tf.data.Iterator.from_string_handle来在两个数据集之间切换

    2. 若tf.data.Dataset不调用repeat函数，对数据集进行迭代，迭代完所有样本，即一个
       epoch，会抛出OutOfRange异常，捕捉此异常可以界定每个epoch的起止

    3. 若使用batch函数，每次迭代会产生batchsize个样本，迭代到最后，不足batchsize个样本，
       则剩余的样本会一并输出，下一次迭代会抛出OutOfRange异常

    4. 若使用tf.data.Dataset.shuffle()函数（类似tf.train.shuffle_batch()函数），buffer_size的大小会影响shuffle的打乱程度，
       同时，一个epoch内，每个样本都会有且只出现一次，不会有某些样本会重复出现的情况.
       shuffle实现原理为：
              一般数据集比较大，无法直接全部载入内存，所以无法一次性shuffle全部数据，只能维护一个固定大小(buffer_size)的buffer,
       取batch数据的时候，是从buffer随机选择batch数据（实际是一个一个取数据组成batch）然后，从磁盘中读取数据填充buffer，从上面的分析可以发现
       buffer_size的大小会影响shuffle的打乱程度，如果buffer_size的大小比全部数据集大小还大，则会得到一个均匀分布的随机性，若
       buffer_size=1, 则相当于没有shuffle。

    5. 函数tf.data.Dataset.prefetch()，使用生产者-消费者模型，维护一个buffe_size大小的buffer，overlap掉读数据以及计算的耗时
       函数tf.data.Dataset.map()参数num_parallel_calls,表示并行处理的样本数

'''


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dataset
import model


BATCHSIZE = 128
NCLASS = 10


handle, train_iterator, val_iterator, iterator = dataset.get_batches_v3('train_mnist.tfrecords',
                                                              'val_mnist.tfrecords',
                                                              BATCHSIZE, NCLASS)
images, labels = iterator.get_next()

# 直接把input pipline 接入计算图
train_out = model.inference(images)
loss = model.losses(train_out, labels)
train_acc = model.accuray(train_out, labels)
train_step = model.train(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

train_handle = sess.run(train_iterator.string_handle())
val_handle = sess.run(val_iterator.string_handle())

num_epoches = 200

for epoch in range(num_epoches):
    sess.run(train_iterator.initializer)
    while True:
        try:
            sess.run(train_step, feed_dict={handle: train_handle})
        except tf.errors.OutOfRangeError:
           break
    total_correct = 0
    total = 0
    sess.run(train_iterator.initializer)
    # 计算训练集的准确率
    while True:
        try:
            tra_batch, tra_lab, acc = sess.run([images, labels, train_acc],
                                               feed_dict={handle: train_handle})
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
            val_batch, val_lab, acc= sess.run([images, labels, train_acc],
                                              feed_dict={handle: val_handle})
            total += val_batch.shape[0]
            total_correct += val_batch.shape[0]*acc

        except tf.errors.OutOfRangeError:
            print('epoch: %s, val acc: %s' % (epoch, total_correct / total))
            break









