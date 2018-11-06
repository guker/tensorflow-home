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
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='x')
y = tf.placeholder(tf.int32, shape=(None, NCLASS), name='y')

# 计算图构建，训练与验证共用同一个计算图
train_out = model.inference(x)

loss = model.losses(train_out, y)
# tensorboard加入观测scalar为loss
tf.summary.scalar('cross_entropy', loss)
train_acc = model.accuray(train_out, y)
# 加入观测标量train_acc, name为acc
tf.summary.scalar('acc', train_acc)
train_step = model.train(loss)


saver = tf.train.Saver()
# 加入集合，便于在加载模型进行预测的时候，直接能取到graph的相应tensor/op
# 一般该tensor/op没有name(如train_op)，可以通过add_to_collection方式便于预测
# 有name的tensor,可以通过tf.get_default_graph().get_tensor_by_name()获取相应的tensor
tf.add_to_collection('predict', train_out)
tf.add_to_collection('x', x)
tf.add_to_collection('y', y)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# 合并所有的观测
merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('./train_log', sess.graph)
test_writer = tf.summary.FileWriter('./test_log')

num_epoches = 5
i = 0
j = 0
for epoch in range(num_epoches):
    # 初始化迭代器，因为迭代器没有设置repeat函数，所以遍历每个epoch的最后，都会抛出OutOfRange异常
    # 捕捉此异常，可以界定epoch的起止
    # 每次初始化迭代器，迭代器都会重新回到epoch的开头
    sess.run(iterator.initializer)
    while True:
        i += 1
        try:
            tra_batch, tra_lab = sess.run([images, labels])
            summary, _ = sess.run([merged, train_step], feed_dict={
                x: tra_batch, y: tra_lab
            })
            train_writer.add_summary(summary, i)
        except tf.errors.OutOfRangeError:
            break
    saver.save(sess, './model', global_step=epoch+1)
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
            total_correct += tra_batch.shape[0] * acc

        except tf.errors.OutOfRangeError:
            print('epoch: %s, train acc: %s' % (epoch, total_correct / total))
            total = 0
            total_correct = 0
            break
    # 计算验证机的准确率
    sess.run(val_iterator.initializer)
    while True:
        j += 1
        try:
            val_batch, val_lab = sess.run([val_images, val_labels])
            summary, acc = sess.run([merged, train_acc], feed_dict={
                x: val_batch, y: val_lab
            })
            total += val_batch.shape[0]
            total_correct += val_batch.shape[0] * acc
            test_writer.add_summary(summary, j)

        except tf.errors.OutOfRangeError:
            print('epoch: %s, val acc: %s' % (epoch, total_correct / total))
            break

train_writer.close()
test_writer.close()
