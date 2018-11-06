import tensorflow as tf
import numpy as np
import dataset
import matplotlib.pyplot as plt

'''
    模型预测有两种方式
    1. 须把模型的结构重新定义一遍，然后载入对应名字的变量，比较繁琐
    2. 从模型文件中将保存的graph中的所有的节点加载到当前的default_graph中，直接引用graph的节点以及tensor
    
'''

iterator = dataset.get_batches_v1('val_mnist.tfrecords', 1, 10, shuffle=False)

image, label = iterator.get_next()


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./model-5.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    # 获取当前计算图
    graph = tf.get_default_graph()
    # 从计算图中获取对应的节点，返回的每个节点都是一个list，表示节点的每个输出
    pred = graph.get_collection('predict')[0]
    # x = graph.get_collection('x')[0]  #等价与 x = graph.get_tensor_by_name('x:0')
    x = graph.get_tensor_by_name('x:0')
    # y = graph.get_collection('y')[0]
    y = graph.get_tensor_by_name('y:0')
    sess.run(iterator.initializer)
    while True:
        try:
           img, lab = sess.run([image, label])
           res = sess.run(pred, feed_dict={x: img, y: lab})
           im = np.reshape(img, [28, 28])
           plt.title(np.argmax(res, axis=-1))
           plt.imshow(im)
           plt.show()
        except tf.errors.OutOfRangeError:
            break




