import argparse
import logging
from sklearn import datasets
from sklearn.cluster import KMeans
import pickle
import os

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

parser = argparse.ArgumentParser(description='kmeans cluster')
parser.add_argument('--operMode', type=str, default='TRAINING', help='The mode component work on')
parser.add_argument('--n_clusters', type=int, default=8, help='The number of clusters to form as well' \
                                                              'as the number of centroids to generate')
parser.add_argument('--initializer', type=str, default='k-means++', help='Method for initialization')
parser.add_argument('--n_init', type=int, default=10, help='Number of time the k-means algorithm will run with' \
                                                           'different centroid seeds')
parser.add_argument('--max_iter', type=int, default=300,
                    help='Maximum number of iterations of the k-means algorithm for a single run')
parser.add_argument('--tol', type=float, default=1e-4,
                    help='Relative tolerance with regards to inertia to declare convergence')
parser.add_argument('--precompute_distances', type=bool, default=False, help='Precompute distances')
parser.add_argument('--random_state', type=int, default=None,
                    help='Determines random number generation for centroid initialization')
parser.add_argument('--n_jobs', type=int, default=1, help='The number of jobs to use for the computation')
parser.add_argument('--aglorithm', type=str, default='auto', help='K-means algorithm to use')

parser.add_argument('--load_model', type=str, default=None, help='The path of model')

args = parser.parse_args()

def main():
    operMode = args.operMode
    logging.info('operMode: {}'.format(operMode))
    iris = datasets.load_iris()
    x, y = iris.data, iris.target

    if operMode == 'TRAINING':
        n_cluster = args.n_clusters
        initializer = args.initializer
        n_init = args.n_init
        max_iter = args.max_iter
        tol = args.tol
        precompute_distances = args.precompute_distances
        random_state = args.random_state
        n_jobs = args.n_jobs
        aglorithm = args.aglorithm
        logging.info('n_clusters: {}\n'
                     'initializer: {}\n'
                     'n_init: {}\n'
                     'max_iter: {}\n'
                     'tol: {}\n'
                     'precompute_distances: {}\n'
                     'random_state: {}\n'
                     'n_jobs: {}\n'
                     'aglorithm: {}'.format(n_cluster, initializer,
                                            n_init, max_iter, tol, precompute_distances, random_state,n_jobs, aglorithm))
        logging.info('training start...')
        try:
            clf = KMeans(n_clusters=n_cluster,init=initializer,n_init=n_init,
                      max_iter=max_iter,tol=tol,precompute_distances=precompute_distances,

                    random_state=random_state,n_jobs=n_jobs,algorithm=aglorithm).fit(x)
        except Exception as e:
            logging.error("Unexpected Error {}".format(e))
            exit(0)
        logging.info('train finished and start save model...')
        with open('kmeans_model.pkl', 'wb') as f:
            pickle.dump(clf, f)
        logging.info("model saved finished!")

    elif operMode == 'PREDICTION':

        logging.info('load_model: {}'.format(args.load_model))
        #获取全局路径
        #abspath = os.path.abspath(args.load_model)

        model_path = os.path.join(args.load_model, "kmeans_model.pkl")
        if not os.path.exists(model_path):
            try:
               raise Exception('model file {} will be loaded not exists!'.format(model_path))
            except Exception as e:
                logging.error('Unexpected Error {}'.format(e))
                exit(0)
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        pred = clf.predict(x)
    else:
        logging.info('sorry, no support!')

if __name__ == '__main__':
    main()
