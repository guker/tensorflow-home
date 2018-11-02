'''

   1. 使用数据可重新初始化迭代器做input pipline
   2. 因为训练集与验证集可以共用同一个可重新初始化迭代器，
      可以不使用feed_dict方式给batch数据，而是直接把input pipline
      接入graph
   3. 边训练边验证
      每个epoch起止可以通过捕捉OutOfRange异常来界定

    测试可重新初始化迭代器的功能
    经测试，得出如下结论：
    1. 训练集数据与验证集数据可以共用一个迭代器，可通过初始化，改变迭代器的指向

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


train_init_op, val_init_op, iterator = dataset.get_batches_v2('train_mnist.tfrecords',
                                                              'val_mnist.tfrecords',
                                                              BATCHSIZE, NCLASS)

'''
   此迭代器可为训练集数据与验证集数据进行迭代，通过重新初始化可以改变迭代器指向的数据集，
   每次通过重新初始化改变迭代器的所指，而且会指向数据集的开头
'''

images, labels = iterator.get_next()
# 直接把input pipline 接入计算图
train_out = model.inference(images)
loss = model.losses(train_out, labels)
train_acc = model.accuray(train_out, labels)
train_step = model.train(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

num_epoches = 200

for epoch in range(num_epoches):
    # 初始化迭代器，因为迭代器没有设置repeat函数，所以遍历每个epoch的最后，都会抛出OutOfRange异常
    # 捕捉此异常，可以界定epoch的起止
    # 每次初始化迭代器，迭代器都会重新回到epoch的开头
    sess.run(train_init_op)
    while True:
        try:
            sess.run(train_step)
        except tf.errors.OutOfRangeError:
           break
    sess.run(train_init_op)
    total_correct = 0
    total = 0
    # 计算训练集的准确率
    while True:
        try:
            tra_batch, tra_lab, acc = sess.run([images, labels, train_acc])
            total += tra_batch.shape[0]
            total_correct += tra_batch.shape[0]*acc

        except tf.errors.OutOfRangeError:
            print('epoch: %s, train acc: %s' % (epoch, total_correct / total))
            total = 0
            total_correct = 0
            break
    # 计算验证机的准确率
    sess.run(val_init_op)
    while True:
        try:
            val_batch, val_lab, acc= sess.run([images, labels, train_acc])
            total += val_batch.shape[0]
            total_correct += val_batch.shape[0]*acc

        except tf.errors.OutOfRangeError:
            print('epoch: %s, val acc: %s' % (epoch, total_correct / total))
            break
            
'''

training_dataset = tf.data.Dataset.range(5)
training_dataset = training_dataset.shuffle(buffer_size=10)
validation_dataset = tf.data.Dataset.range(3)


iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

sess = tf.Session()

for _ in range(20):
   sess.run(training_init_op)
   while True:
       try:
          print(sess.run(next_element))
       except tf.errors.OutOfRangeError:
          #pass
          break

   sess.run(validation_init_op)
   while True:
       try:
          sess.run(next_element)
       except tf.errors.OutOfRangeError:
          #pass
          break
'''





