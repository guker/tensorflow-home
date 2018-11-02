'''

   1. 使用数据 one shot iterator 做input pipline
   2. 训练集与验证集使用不同的迭代器，input pipline不能直接
      接入计算图，所以只能使用feed_dict方式给数据

   3. 边训练边验证
      通过指定每个epcoh内的迭代次数来界定epoch的边界

   note:
       计算图里面有update(优化更新)以及state(状态)
       Session.run(fetchs)函数，若fetchs为一list，比如session.run([train_op, acc])
       是先更新权重还是先计算acc？
       a = tf.Variable(1)
       b = tf.Variable(2)
       c = a + b
       assign = tf.assign(a, 5)

       sess = tf.Session()
       sess.run(tf.global_variable_initializer())

       print(sess.run([c, assign])) vs print(see.run([assign, c]))
       两个结果不同，前者为[3, 5], 后者为[5, 7]

       结论是，不要写成列表，改成单步

       需要tf.control_dependencies()控制依赖

   验证one shot迭代器功能

   1： Dataset的repeat()函数会将input数据以一个有限次进行重复，数据的每次重复为一个epoch
      若shuffle转换之前应用repeat转换，那么epoch的边界是模糊的，防止过拟合，也就是说，特定的元素即使在其他只出现一次之前
      可以被重复， 如果在repeat转换之前使用shuffle转换，那么每个epoch的边界可以保证，但是每个epoch的开始阶段性能会降低
      这与shuffle内部实现有关，一般推荐shuffle->repeat, 可以保证顺序，也能考虑到每个epoch的随机性

   2. one_shot迭代器，一旦抛出OutOfRange，则没有办法再次使用one shot迭代器重新指向数据集的开头，所以只能repeat数据集，否则无法
      进行循环训练


'''


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dataset
import model


BATCHSIZE = 128
NCLASS = 10

EPOCHES = 200


train_iterator, val_iterator = dataset.get_batches_v4('train_mnist.tfrecords',
                                                      'val_mnist.tfrecords',
                                                       BATCHSIZE, NCLASS, EPOCHES)
images, labels = train_iterator.get_next()
val_images, val_labels = val_iterator.get_next()


x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.float32, shape=(None, NCLASS))
train_out = model.inference(x)
loss = model.losses(train_out, y)
train_acc = model.accuray(train_out, y)
train_step = model.train(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
'''
    这里需要指定每个epoch里面有多少个step，因为使用repeat函数后，每个epoch没有明确的边界
    跟mxnet的方式类似,需要指定epoch内的迭代次数
    训练集的样本总数可以知道，因为里面做了shuffle，模型总有机会见到所有的样本
    验证集必须知道样本数，这样才能清楚界定每个epoch的边界
'''
train_step_in_epoch = 400
val_step_in_epoch = 79  # 10000//batchsize

for epoch in range(EPOCHES):

    total_correct = 0
    total = 0
    for _ in range(train_step_in_epoch):

        img_batch, lab_batch = sess.run([images, labels])

        _, acc = sess.run([train_step,train_acc], feed_dict={
                         x: img_batch,
                         y: lab_batch})
        total += img_batch.shape[0]
        total_correct += img_batch.shape[0]*acc
    print('epoch: %s, train acc: %s' % (epoch, total_correct / total))

    total_correct = 0
    total = 0

    # 计算验证机的准确率
    for _ in range(val_step_in_epoch):

        img_batch, lab_batch = sess.run([val_images, val_labels])

        acc = sess.run(train_acc, feed_dict={
                          x: img_batch,
                          y: lab_batch})
        total += img_batch.shape[0]
        total_correct += img_batch.shape[0]*acc

    print('epoch: %s, val acc: %s' % (epoch, total_correct / total))










