import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def cosine_distance(X, Y):
    '''

    Cosine distance is defined as 1.0 minus the cosine similarity

    '''
    X_norm = tf.sqrt(tf.reduce_sum(tf.square(X), axis=1))
    Y_norm = tf.sqrt(tf.reduce_sum(tf.square(Y), axis=1))
    XY_norm = tf.multiply(X_norm, tf.expand_dims(Y_norm, 1))
    XY = tf.multiply(X, Y[:, None])
    XY = tf.reduce_sum(XY, 2)
    similarity = XY / XY_norm
    distance = 1 - similarity
    return distance


def euclidean_distance(X, Y):
    '''

     欧式距离

    '''
    distance = tf.norm(tf.subtract(X, tf.expand_dims(Y, 1)), axis=2)
    return distance


def manhattan_distance(X, Y):
    '''

    曼哈顿距离

    '''
    distance = tf.reduce_sum(tf.abs(tf.subtract(X, tf.expand_dims(Y, 1))), axis=2)
    return distance


DISTANCE_FUNCTIONS = {
    'euclidean': euclidean_distance,
    'l2': euclidean_distance,
    'manhattan': manhattan_distance,
    'l1': manhattan_distance,
    'cosine': cosine_distance
     }

class KNeighborsClassifier(object):
     """
     classifier implementing k-nearest neighbors
     """
     def __init__(self, n_neighbors=5, metric='euclidean', batch_size=128):
         self.n_neighbors = n_neighbors
         self.metric = metric
         self.batch_size = batch_size
         self._fit_X = None
         self._fit_y = None
         self._input_dim = None
         self._n_classes = None
         self._predict_proda = False

         if not isinstance(n_neighbors, int):
             raise TypeError('n neighbors must be an integer')
         if metric not in DISTANCE_FUNCTIONS.keys():
             raise TypeError('unrecognized metric: {}'.format(metric))

     def get_params(self):
         return {
             'metric': self.metric,
             'n_neoghbors': self.n_neighbors,
             'batch_size': self.batch_size
         }

     def _check_x(self, x):
         if not hasattr(x, 'shape') or len(x.shape) !=2:
             raise TypeError('X must be a numpy array, shape [n_samples, n_features]')
         x = x.astype('float32')
         return x

     def _check_y(self, y):
         if isinstance(y, list):
             y = np.array(y, dtype=int)
         if len(y.shape) >= 2:
             raise TypeError('y must be a numpy array, shape[n_samples]')

         y = np.array(y, dtype=int)
         self._n_samples = max(y.shape)
         y = y.reshape(self._n_samples)

         self._n_classes = len(np.unique(y))
         _y = np.zeros((self._n_samples, self._n_classes))
         _y[np.arange(self._n_samples), y] = 1
         return _y

     def fit(self, X, y):
         """
         X: train data, a numpy array, shape [n_samples, n_features]
         y: target values of numpy array with shape [n_samples]
         """

         X = self._check_x(X)
         y = self._check_y(y)

         assert X.shape[0] == self._n_samples, "samples in X not equal to smaples in y," \
                                               " got {} != {}".format(X.shape[0], self._n_samples)

         self._input_dim = X.shape[1]
         self._fit_X = X
         self._fit_y = y


     def _predict(self, X):
         distance = DISTANCE_FUNCTIONS[self.metric](self._fit_X, X)
         top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=self.n_neighbors)
         prediction_indices = tf.gather(self._fit_y, top_k_indices)

         count_of_predictions = tf.reduce_sum(prediction_indices, axis=1)
         prediction = tf.argmax(count_of_predictions, axis=1)
         if self._predict_proba:
             proba = tf.div(count_of_predictions, self.n_neighbors)
             return proba
         return prediction

     def _predict_proba(self, X):
         self._predict_proba = True
         y_pred = self._predict(X)
         self._predict_proba = False
         return y_pred

     def predict(self, X):
         X =self._check_x(X)
         dataset = tf.data.Dataset.from_tensor_slices((X))
         dataset = dataset.batch(self.batch_size)
         iterator = dataset.make_one_shot_iterator()
         x_batch = iterator.get_next()
         y_pred = self._predict(x_batch)

         sess = tf.Session()
         with sess.as_default():
             init = tf.global_variables_initializer()
             sess.run(init)
             output = []
             while True:
                 try:
                     _y_pred =sess.run(y_pred)
                     output.extend(_y_pred)
                     print(_y_pred)
                 except tf.errors.OutOfRangeError:
                     break
         sess.close()
         return np.array(output)


if __name__ == '__main__':

    mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
    x_train, y_train_dense = mnist.train.next_batch(50000)
    x_test, y_test_dense = mnist.test.next_batch(1000)
    y_train = np.argmax(y_train_dense, axis=1)
    #print(y_train)

    y_test = np.argmax(y_test_dense, axis=1)
    #print(y_test)

    knn = KNeighborsClassifier(n_neighbors=7,batch_size=1)
    knn.fit(x_train, y_train)

    res = knn.predict(x_test)
    #print(res)
    accuray = 0
    for i in range(len(y_test)):
        print('Test', i ,'Precetion:', np.argmax(res[i]),
              'True Class:', y_test[i])
        if np.argmax(res[i]) == y_test[i]:
            accuray +=1./len(x_test)

print('Accuary:', accuray)










