#tensorflow的feed_dict
#需要每次迭代生成一个batch

#Dataset

class DataSet（object）：
    def  _init__(self,
                 images,
                 labels,......):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0 # 已经过了多少个epoch
        self._index_in_epoch = 0
        self._num_examples



    def next_batch(self, batchsize, fake_data=False, shuffle=True):
        start = self._index_in_epoch
        # 第一个epoch需要shuffle
        if self._epochs_completed ==0 and start ==0 and shuffle:
           perm = numpy.arange(self._num_examples)
           numpy.random.shuffle(perm)
           self._images = self._images[perm]
           self._labels = self._labels[perm]
        # go to the next epoch

        if start + batchsize > self._num_examples:
           self._epochs_completed += 1
           rest_num_examples = self._num_examples - start
           images_rest_part = self._images[start:self._num_examples]
           labels_rest_part = self._images[start:self._num_examples]
           if shuffle:
              perm = numpy.arange(self._num_examples)
              self._images = self._images[perm]
              self._labels = self._lables[perm]

           start = 0
           self._index_in_epoch = batchsize - rest_num_examples
           images_new_part = self._images[start:end]
           labels_new_part = self._labels[start:end]
           return numpy.concatenate((images_rest_part, images_new_part), axis=0), numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
           self._index_in_epoch += batchsize
           end = self._index_in_epoch
           return self._images[start:end], self._labels[start:end]
