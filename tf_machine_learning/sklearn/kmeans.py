import argparse
import logging
from sklearn.cluster import KMeans
import pickle
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import json

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

parser = argparse.ArgumentParser(description='kmeans cluster')

parser.add_argument('--operMode', type=str, default='PREDICTION',
                    help='The mode component work on')

parser.add_argument('--n_clusters', type=int, default=3,
                    help='The number of clusters to form as well' \
                         'as the number of centroids to generate')

parser.add_argument('--initializer', type=str, default='k-means++',
                    help='Method for initialization')

parser.add_argument('--n_init', type=int, default=10,
                    help='Number of time the k-means algorithm will run with different centroid seeds')

parser.add_argument('--max_iter', type=int, default=100,
                    help='Maximum number of iterations of the k-means algorithm for a single run')

parser.add_argument('--tol', type=float, default=1e-4,
                    help='Relative tolerance with regards to inertia to declare convergence')

parser.add_argument('--precompute_distances', type=bool, default=False,
                    help='Precompute distances')

parser.add_argument('--random_state', type=int, default=None,
                    help='Determines random number generation for centroid initialization')

parser.add_argument('--n_jobs', type=int, default=1,
                    help='The number of jobs to use for the computation')

parser.add_argument('--aglorithm', type=str, default='auto',
                    help='K-means algorithm to use')

parser.add_argument('--has_label', type=bool, default=True,
                    help='data has label or not')

parser.add_argument('--label_name', type=str, default='species',
                    help='label col name')

parser.add_argument('--load_model', type=str, default='.',
                    help='The path of model')

args = parser.parse_args()


def main():
    operMode = args.operMode
    logging.info('operMode: {}'.format(operMode))
    '''
    iris.csv
    sepal_length,sepal_width,petal_length,petal_width,species
    5.1,3.5,1.4,0.2,setosa
    '''
    input_file = 'iris.csv'
    df = pd.read_csv(input_file)

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
        has_label = args.has_label
        label_name = args.label_name
        if has_label:
            if label_name is None:
                try:
                    raise Exception('if parameter has_label is true, label_name must not be none')
                except Exception as e:
                    logging.error(e)
                    exit(0)

        logging.info('model parameter as follow:\n'
                     'n_clusters: {}\n'
                     'initializer: {}\n'
                     'n_init: {}\n'
                     'max_iter: {}\n'
                     'tol: {}\n'
                     'precompute_distances: {}\n'
                     'random_state: {}\n'
                     'n_jobs: {}\n'
                     'aglorithm: {}\n'
                     'has_label: {}\n'
                     'label_name: {}'.format(n_cluster, initializer,
                                             n_init, max_iter, tol, precompute_distances, random_state, n_jobs,
                                             aglorithm, has_label, label_name))
        if has_label:
            columns = df.columns.tolist()
            targets = df[label_name].values
            columns.remove(label_name)
            featrues = df[columns].values
        else:
            featrues = df.values

        logging.info('training start...')
        try:
            clf = KMeans(n_clusters=n_cluster, init=initializer, n_init=n_init,
                         max_iter=max_iter, tol=tol, precompute_distances=precompute_distances,

                         random_state=random_state, n_jobs=n_jobs, algorithm=aglorithm).fit(featrues)
        except Exception as e:
            logging.error("Unexpected Error {}".format(e))
            exit(0)
        logging.info('train finished and start save model...')
        with open(os.path.join('.', 'kmans_model.pkl'), 'wb') as f:
            pickle.dump(clf, f)
        logging.info("model saved finished!")

        pfmn_dict = {}
        pfmn_dict['evaluation'] = []
        if has_label:
            ari = metrics.adjusted_rand_score(targets, clf.labels_)
            ami = metrics.adjusted_mutual_info_score(targets, clf.labels_)
            homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(targets, clf.labels_)
            fmi = metrics.fowlkes_mallows_score(targets, clf.labels_)
            pfmn_dict['evaluation'].append({'name': '调整兰德系数', 'value': ari})
            pfmn_dict['evaluation'].append({'name': '调整互信息系数', 'value': ami})
            pfmn_dict['evaluation'].append({'name': '同质性', 'value': homogeneity})
            pfmn_dict['evaluation'].append({'name': '完整性', 'value': completeness})
            pfmn_dict['evaluation'].append({'name': 'v_measure', 'value': v_measure})
            pfmn_dict['evaluation'].append({'name': 'Fowlkes_Mallows Index', 'value': fmi})
        silhouette = metrics.silhouette_score(featrues, clf.labels_)
        calinskiharabaz = metrics.calinski_harabaz_score(featrues, clf.labels_)
        pfmn_dict['evaluation'].append({'name': '轮廓系数', 'value': silhouette})
        pfmn_dict['evaluation'].append({'name': 'Calinski-Harabaz Index', 'value': calinskiharabaz})

        pfm_json = json.dumps(pfmn_dict)
        with open(os.path.join('.', 'performance.json'), 'w') as f:
            f.write(pfm_json)
        logging.info('evaluation finished!')

    elif operMode == 'PREDICTION':
        logging.info('load_model: {}'.format(args.load_model))

        has_label = args.has_label
        label_name = args.label_name
        if has_label:
            if label_name is None:
                try:
                    raise Exception('if parameter has_label is true, label_name must not be none')
                except Exception as e:
                    logging.error(e)
                    exit(0)
        if has_label:
            columns = df.columns.tolist()
            targets = df[label_name].values
            columns.remove(label_name)
            featrues = df[columns].values
        else:
            featrues = df.values

        model_path = args.load_model + '/kmeans_model.pkl'
        if not os.path.exists(model_path):
            try:
                raise Exception('model file {} will be loaded not exists!'.format(model_path))
            except Exception as e:
                logging.error('Unexpected Error {}'.format(e))
                exit(0)
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        pred = clf.predict(featrues)
        df['prediction'] = pred
        df.to_csv(os.path.join('.', 'res.csv'), index=False)

        pfmn_dict = {}
        pfmn_dict['evaluation'] = []
        if has_label:
            ari = metrics.adjusted_rand_score(targets, pred)
            ami = metrics.adjusted_mutual_info_score(targets, pred)
            homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(targets, pred)
            fmi = metrics.fowlkes_mallows_score(targets, pred)
            pfmn_dict['evaluation'].append({'name': '调整兰德系数', 'value': ari})
            pfmn_dict['evaluation'].append({'name': '调整互信息系数', 'value': ami})
            pfmn_dict['evaluation'].append({'name': '同质性', 'value': homogeneity})
            pfmn_dict['evaluation'].append({'name': '完整性', 'value': completeness})
            pfmn_dict['evaluation'].append({'name': 'v_measure', 'value': v_measure})
            pfmn_dict['evaluation'].append({'name': 'Fowlkes_Mallows Index', 'value': fmi})
        silhouette = metrics.silhouette_score(featrues, pred)
        calinskiharabaz = metrics.calinski_harabaz_score(featrues, pred)
        pfmn_dict['evaluation'].append({'name': '轮廓系数', 'value': silhouette})
        pfmn_dict['evaluation'].append({'name': 'Calinski-Harabaz Index', 'value': calinskiharabaz})

        pfm_json = json.dumps(pfmn_dict)
        with open(os.path.join('.', 'performance.json'), 'w') as f:
            f.write(pfm_json)
        logging.info('evaluation finished!')

    else:
        logging.info('sorry, no support!')


if __name__ == '__main__':
    main()
