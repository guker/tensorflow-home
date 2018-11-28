# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import keras
import cv2
import json
import matplotlib
from matplotlib import pyplot as plt
import datetime
import sklearn
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, average_precision_score

flags = tf.app.flags

# ======================================common=================================
flags.DEFINE_string('component_type', 'untrainable', 'Component status')

# ========================untrainable model input==============================
flags.DEFINE_string('input_file', None, 'Train set path list')
flags.DEFINE_string('split_mode', 'shuffle', 'Split dataset into train set and validation set')
flags.DEFINE_float('split_ratio', 0.8, 'Split proportion')

# =========================untrainable model super parameter===================
flags.DEFINE_integer('batch_size', 64, 'Training batch size')
flags.DEFINE_integer('num_classes', 2, 'Number of classes')
flags.DEFINE_integer('num_epochs', 1, 'Number of epoch')
flags.DEFINE_float('initial_lr', 0.01, 'Initial learning rate')
flags.DEFINE_string('lr_schedule', 'time_base', 'Schedule mode about learing rate')
flags.DEFINE_string('optimizer', 'SGD', 'Optimizer method')
flags.DEFINE_string('initial_weight', 'glorot_normal', 'Layers weight initializers')
flags.DEFINE_string('initial_bias', 'Zeros', 'Layers bias initializers')
flags.DEFINE_string('keep_prob', '0.5,0.5', 'Dropout rate')
flags.DEFINE_string('loss', 'categorical_crossentropy', ' Loss  function')
flags.DEFINE_string('metrics', 'accuracy', 'Metrices function')

# =========================untrainable model output===============================
flags.DEFINE_string('model_path', None, 'Store model path')
flags.DEFINE_string('log_dir', None, 'Store log file')

# ========================fine tuning model input=================================
flags.DEFINE_string('input_model', None, 'Reload model path')
# =========================fine tuning model super parameter======================
flags.DEFINE_string('trainable_layers', 'fc8,fc7', 'Trainable layers name')
# =========================predict output layer===================================
flags.DEFINE_string('output_layer', 'fc6', 'output single layer name')

FLAGS = flags.FLAGS

'''
数据生成器
'''


def datasplit(datalist, ratio=0.8, shuffle=False):
    '''
    shuffle the whole dataset , split to train set and  validation set
    '''
    filelist = []
    with open(datalist) as f:
        for line in f.readlines():
            filelist.append(line.strip())
    if shuffle:
        np.random.shuffle(filelist)
    num_train = int(len(filelist) * ratio)
    train_set = filelist[:num_train]
    validation_set = filelist[num_train:]
    return train_set, validation_set


def datagenerator(filelist, batch_size=32, shuffle=False):
    '''
    data generator by yeild
    '''

    def load_images(img_batch, batch_size):
        images = np.ndarray([batch_size, 227, 227, 3])
        labels = np.zeros([batch_size, 2], np.int32)

        for i in range(len(img_batch)):
            items = img_batch[i].split(' ')
            img = cv2.imread(items[0])

            img = cv2.resize(img, (227, 227))

            img = img.astype(np.float32)
            label = int(items[1])
            images[i] = img
            labels[i][label] = 1
        return images, labels

    while True:
        if shuffle:
            np.random.shuffle(filelist)
        filenames = np.array(filelist)

        for i in range(np.ceil(1.0 * len(filenames) // batch_size).astype(int)):
            batch_img = filenames[i * batch_size:(i + 1) * batch_size]
            images, labels = load_images(batch_img, batch_size)
            yield images, labels


def pred_generator(filelist, batch_size=32):
    def load_image(img_batch, batch_size):
        images = np.ndarray([batch_size, 227, 227, 3])
        for i in range(len(img_batch)):
            items = img_batch[i].split(' ')
            img = cv2.imread(items[0])
            img = cv2.resize(img, (227, 227))
            img = img.astype(np.float32)
            images[i] = img
        return images

    while True:
        for i in range(np.ceil(1.0 * len(filelist) // batch_size).astype(int)):
            batch_img = filelist[i * batch_size:(i + 1) * batch_size]
            imgs = load_image(batch_img, batch_size)
            yield imgs


def get_labels(filelist):
    labels = []
    for i in range(len(filelist)):
        items = filelist[i].split(' ')
        labels.append(int(items[1]))
    return labels


'''
模型构建
'''


class LRN(keras.engine.Layer):
    def __init__(self, depth_radius=2, bias=1.0, alpha=1, beta=1, **kwargs):
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
        super(LRN, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LRN, self).build(input_shape)

    def call(self, x):
        return tf.nn.lrn(x, depth_radius=self.depth_radius, bias=self.bias, alpha=self.alpha, beta=self.beta)

    def compute_output_shape(self, input_shape):
        return input_shape


class AlexNet():
    def __init__(self, input_shape=(227, 227, 3), classes=1000, keep_prob=[0.5, 0.5],
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros'):
        self.input_shape = input_shape
        self.num_classes = classes
        self.keep_prob = keep_prob
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.build_model()

    def build_model(self):
        self.input = keras.layers.Input(self.input_shape)

        self.conv1 = keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='VALID',
                                         kernel_initializer=self.kernel_initializer, activation='relu',
                                         bias_initializer=self.bias_initializer, name='conv1')(self.input)
        self.bn1 = keras.layers.BatchNormalization(axis=3, name='bn1')(self.conv1)
        # self.lrn1 = LRN(depth_radius=2,bias=1e-04,alpha=2e-05,beta=0.75)(self.conv1)
        self.pool1 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='VALID', name='pool1')(self.bn1)

        self.conv2 = keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='SAME',
                                         kernel_initializer=self.kernel_initializer, activation='relu',
                                         bias_initializer=self.bias_initializer, name='conv2')(self.pool1)
        self.bn2 = keras.layers.BatchNormalization(axis=3, name='bn2')(self.conv2)
        # self.lrn2 = LRN(depth_radius=2,bias=1e-04,alpha=2e-05,beta=0.75)(self.conv2)
        self.pool2 = keras.layers.MaxPool2D(pool_size=(3, 3,), strides=(2, 2), padding='VALID', name='pool2')(self.bn2)

        self.conv3 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                         kernel_initializer=self.kernel_initializer, activation='relu',
                                         bias_initializer=self.bias_initializer, name='conv3')(self.pool2)

        self.conv4 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                         kernel_initializer=self.kernel_initializer, activation='relu',
                                         bias_initializer=self.bias_initializer, name='conv4')(self.conv3)

        self.conv5 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                         kernel_initializer=self.kernel_initializer, activation='relu',
                                         bias_initializer=self.bias_initializer, name='conv5')(self.conv4)
        self.pool5 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='VALID', name='pool5')(self.conv5)

        self.flat = keras.layers.Flatten()(self.pool5)
        self.fc1 = keras.layers.Dense(units=4096, activation='relu', kernel_initializer=self.kernel_initializer,
                                      bias_initializer=self.bias_initializer, name='fc6')(self.flat)

        self.dropout1 = keras.layers.Dropout(rate=self.keep_prob[0])(self.fc1)

        self.fc2 = keras.layers.Dense(units=4096, activation='relu', kernel_initializer=self.kernel_initializer,
                                      bias_initializer=self.bias_initializer, name='fc7')(self.dropout1)

        self.dropout2 = keras.layers.Dropout(rate=self.keep_prob[1])(self.fc2)

        self.fc3 = keras.layers.Dense(units=self.num_classes, activation='softmax',
                                      kernel_initializer=self.kernel_initializer,
                                      bias_initializer=self.bias_initializer, name='fc8')(self.dropout2)

        self.model = keras.models.Model(inputs=self.input, outputs=self.fc3, name='alexnet')


def get_optimizers(optimizer, learning_rate=0.01):
    if optimizer == 'SGD':
        return keras.optimizers.SGD(lr=learning_rate)
    if optimizer == 'Adadelta':
        return keras.optimizers.Adadelta(lr=learning_rate)
    if optimizer == 'Adagrad':
        return keras.optimizers.Adagrad(lr=learning_rate)
    if optimizer == 'RMSProp':
        return keras.optimizers.RMSprop(lr=learning_rate)
    if optimizer == 'Adam':
        return keras.optimizers.Adam(lr=learning_rate)


def top_k(y_true, y_pred, k=1):
    nrows = y_pred.shape[0]
    if k > y_pred.shape[1]:
        return None

    pred_correct = np.zeros([nrows],dtype=np.int32)

    for i in range(nrows):
        target = y_true[i]
        pred_score = y_pred[i]
        top_k_idx = pred_score.argsort()[-k:][::-1]
        if target in top_k_idx:
            pred_correct[i] = 1
        else:
            pred_correct[i] = 0
    return np.mean(pred_correct)





def main(unused_argv):
    '''
     this code has three mode, untrainable, finetuning, prediction
    '''
    component_type = FLAGS.component_type
    if component_type == 'untrainable':
        input_file = FLAGS.input_file
        split_mode = FLAGS.split_mode
        split_ratio = FLAGS.split_ratio
        # 划分训练集验证集
        if split_mode == 'shuffle':
            train_set, validation_set = datasplit(input_file, split_ratio, shuffle=True)
        else:
            train_set, validation_set = datasplit(input_file, split_ratio)
        # 生成迭代器
        batch_size = FLAGS.batch_size
        train_generator = datagenerator(train_set, batch_size=batch_size, shuffle=True)
        validation_generator = datagenerator(validation_set, batch_size=batch_size)

        # 构建模型
        initial_weight = FLAGS.initial_weight
        initial_bias = FLAGS.initial_bias
        keep_prob = [float(i) for i in FLAGS.keep_prob.split(',')]
        model = AlexNet(classes=FLAGS.num_classes, kernel_initializer=initial_weight,
                        keep_prob=keep_prob, bias_initializer=initial_bias).model

        # 设置优化器
        opt = FLAGS.optimizer
        initial_learning = FLAGS.initial_lr
        optimizer = get_optimizers(opt, initial_learning)

        model.compile(optimizer=optimizer, loss=FLAGS.loss, metrics=[FLAGS.metrics])
        # 设置日志以tensorboard
        tensorboard_path = FLAGS.log_dir
        tb_cb = keras.callbacks.TensorBoard(log_dir=tensorboard_path,
                                            write_graph=True,
                                            write_images=True)
        # 开始训练
        num_epochs = FLAGS.num_epochs
        steps_per_epoch = (len(train_set) // batch_size)
        validation_steps = (len(validation_set) // batch_size)

        model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                            verbose=1, callbacks=[tb_cb], validation_data=validation_generator,
                            validation_steps=validation_steps)

        # 保存模型
        model_path = FLAGS.model_path
        if model_path.endswith('/') or model_path.endswith('\\'):
            model_name = model_path + 'model_' + str(datetime.datetime.now().date()).replace('-', '') + '_' + \
                         str(datetime.datetime.now().time()).replace(':', '').split('.')[0] + '.h5'
        else:
            model_name = model_path + '/model_' + str(datetime.datetime.now().date()).replace('-', '') + '_' + \
                         str(datetime.datetime.now().time()).replace(':', '').split('.')[0] + '.h5'
        model.save(model_name)

        # 评估模型,使用验证集评估模型
        eval_val_generator = pred_generator(validation_set, batch_size=1)
        y_true = get_labels(validation_set)
        y_pred_prob = model.predict_generator(eval_val_generator, steps=len(validation_set))

        y_pred_target = np.argmax(y_pred_prob, axis=1)

        cfmt = confusion_matrix(y_true, y_pred_target)  # 计算混淆矩阵
        fprs = []
        tprs = []
        mauc = 0.0 # mauc
        map = 0.0 # map
        acc_top1 = top_k(y_true,y_pred_prob,k=1)
        acc_top5 = top_k(y_true,y_pred_prob,k=5)
        for c in range(FLAGS.num_classes):
            y_true_binary = [int(i == c) for i in y_true]
            y_pred_binary = y_pred_prob[:, c]
            fpr, tpr, thres = roc_curve(y_true_binary, y_pred_binary, pos_label=1)
            AUC = roc_auc_score(y_true_binary, y_pred_binary)
            ap = average_precision_score(y_true_binary, y_pred_binary)
            fprs.append(fpr.tolist())
            tprs.append(tpr.tolist())
            mauc += AUC
            map += ap
        mauc = 1.0 * mauc / FLAGS.num_classes
        map = 1.0 * map / FLAGS.num_classes
        result = {'fpr':fprs,
                  'tprs':tprs,
                  'cfmt':cfmt.tolist(),
                  'mauc':mauc,
                  'map':map,
                  'top-1':acc_top1,
                  'top-5':acc_top5}
        str_res = json.dumps(result)
        with open('preformances.json','w') as f:
            f.write(str_res)

    elif component_type == 'fine-tuning':
        input_file = FLAGS.input_file
        split_mode = FLAGS.split_mode
        split_ratio = FLAGS.split_ratio
        # 划分训练集与验证集
        if split_mode == 'shuffle':
            train_set, validation_set = datasplit(input_file, split_ratio, shuffle=True)
        else:
            train_set, validation_set = datasplit(input_file, split_ratio)

        # 生成迭代器
        batch_size = FLAGS.batch_size
        train_generator = datagenerator(train_set, batch_size=batch_size, shuffle=True)
        validation_generator = datagenerator(validation_set, batch_size=batch_size)

        # 构建模型
        premodel_path = '/Users/qwhu/github/alexnet/model_20181127_215511.h5'
        model = keras.models.load_model(premodel_path, compile=False)

        trainable_layers = FLAGS.trainable_layers
        trainable_layers_list = [str(i) for i in trainable_layers.split(',')]
        for layer in model.layers:
            if layer.name not in trainable_layers_list:
                layer.trainable = False
        for layer in model.layers:
            print(layer.name, layer.trainable)
        # 设置优化器
        opt = FLAGS.optimizer
        initial_learning = FLAGS.initial_lr
        optimizer = get_optimizers(opt, initial_learning)

        model.compile(optimizer=optimizer, loss=FLAGS.loss, metrics=[FLAGS.metrics])
        # model.summary()

        # 设置日志以tensorboard
        tensorboard_path = FLAGS.log_dir
        tb_cb = keras.callbacks.TensorBoard(log_dir=tensorboard_path,
                                            write_graph=True,
                                            write_images=True)
        # 开始训练
        num_epochs = FLAGS.num_epochs
        steps_per_epoch = (len(train_set) // batch_size)
        validation_steps = (len(validation_set) // batch_size)

        model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                            verbose=1, callbacks=[tb_cb], validation_data=validation_generator,
                            validation_steps=validation_steps)

        # 保存模型
        model_path = FLAGS.model_path
        if model_path.endswith('/') or model_path.endswith('\\'):
            model_name = model_path + 'model_' + str(datetime.datetime.now().date()).replace('-', '') + '_' + \
                         str(datetime.datetime.now().time()).replace(':', '').split('.')[0] + '.h5'
        else:
            model_name = model_path + '/model_' + str(datetime.datetime.now().date()).replace('-', '') + '_' + \
                         str(datetime.datetime.now().time()).replace(':', '').split('.')[0] + '.h5'
        model.save(model_name)

        # 评估模型
        eval_val_generator = pred_generator(validation_set, batch_size=1)
        y_true = get_labels(validation_set)
        y_pred_prob = model.predict_generator(eval_val_generator, steps=len(validation_set))

        y_pred_target = np.argmax(y_pred_prob, axis=1)

        cfmt = confusion_matrix(y_true, y_pred_target)  # 计算混淆矩阵

        fprs = []
        tprs = []
        mauc = 0.0  # mauc
        map = 0.0 # map
        acc_top1 = top_k(y_true,y_pred_prob,k=1)
        acc_top5 = top_k(y_true,y_pred_prob,k=5)
        for c in range(FLAGS.num_classes):
            y_true_binary = [int(i == c) for i in y_true]
            y_pred_binary = y_pred_prob[:, c]
            fpr, tpr, thres = roc_curve(y_true_binary, y_pred_binary, pos_label=1)
            AUC = roc_auc_score(y_true_binary, y_pred_binary)
            ap = average_precision_score(y_true_binary, y_pred_binary)
            fprs.append(fpr.tolist())
            tprs.append(tpr.tolist())
            mauc += AUC
            map += ap
        mauc = 1.0 * mauc / FLAGS.num_classes
        map = 1.0 * map / FLAGS.num_classes
        result = {'fprs':fprs,
                  'tprs':tprs,
                  'cfmt':cfmt.tolist(),
                  'mauc':mauc,
                  'map':map,
                  'top-1':acc_top1,
                  'top-5':acc_top5}
        str_res = json.dumps(result)
        with open('performaces.json','w') as f:
            f.write(str_res)

    elif component_type == 'prediction':
        input_file = FLAGS.input_file
        test_set, _ = datasplit(input_file, 1.0)

        # 生成迭代器
        pred_test_genertaor = pred_generator(test_set, batch_size=1)
        y_true = get_labels(test_set)

        # 构建模型
        premodel_path = '/Users/qwhu/github/alexnet/model_20181127_215511.h5'
        model = keras.models.load_model(premodel_path, compile=False)

        output_layer = FLAGS.output_layer
        intermediate_layer_model = keras.models.Model(inputs=model.input,
                                                      outputs=model.get_layer(output_layer).output)
        y_pred = intermediate_layer_model.predict_generator(pred_test_genertaor, steps=len(test_set))


    else:
        print('sorry, no support!')


if __name__ == '__main__':
    tf.app.run()
