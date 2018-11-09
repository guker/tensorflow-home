import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datagenerator import ImageDataGenerator
import argparse

'''
    使用imagnet alexnet的模型作为预训练模型进行kaggle的猫狗识别
    
'''

parser = argparse.ArgumentParser()
parser.add_argument('--testfile', type=str, default='./val.txt', help="test file list")
parser.add_argument('--batch_size', type=int, default=1, help='number of exmaples in one iterator')
parser.add_argument('--checkpoint_path', action='store', type=str,
                    default='./checkpoint', help='checkpoint_path')
parser.add_argument('--top_N', action='store', type=int,
                    default=1, help='whether the targets are in the top K predictions.')

args = parser.parse_args()
batchsize = args.batch_size
checkpointpath = args.checkpoint_path
checkpointpath = checkpointpath + '/model_epoch2.ckpt.meta'

test_generator = ImageDataGenerator(args.testfile, horizontal_flip=False,
                                    shuffle=False)

test_batches_per_epoch = np.floor(test_generator.data_size /
                                  args.batch_size).astype(np.int16)

with tf.Session() as sess:
    # 导入计算图，作为当前默认图
    saver = tf.train.import_meta_graph(checkpointpath)
    # 导入模型参数
    saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_path))
    # 获取当前导入的计算图
    graph = tf.get_default_graph()
    # 根据名字获取计算图中的节点(tensor或者op)
    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name('y:0')
    keep_prob = graph.get_tensor_by_name('kp:0')

    output = graph.get_tensor_by_name('fc8/fc8:0') # 这里只能获取有name的tensor，即fc函数中act

    step = 1
    while step < test_batches_per_epoch:
        batch_xs, batch_ys = test_generator.next_batch(args.batch_size)
        out = sess.run(output, feed_dict={
            x: batch_xs,
            y: batch_ys,
            keep_prob: 1.0
        })
        single_images = np.reshape(batch_xs, [227, 227, 3])
        single_images[:, :, 0] += np.array([104])
        single_images[:, :, 1] += np.array([117])
        single_images[:, :, 2] += np.array([124])
        single_images = single_images.astype('uint8')
        '''
        1. 图片的uint8编码，如果使用init8会出现负数，显示存在问题
        2. cv2读取图片为BGR通道，而imshow的显示为RGB通道，需要调换通道，否则会出错
        '''

        #single_images[:, :, [0, 2]] = single_images[:, :, [2, 0]]
        single_images = single_images[..., ::-1]
        plt.imshow(single_images)
        plt.title(np.argmax(out, axis=-1))

        plt.show()
        step += 1
