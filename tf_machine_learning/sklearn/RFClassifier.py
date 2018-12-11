import os
import argparse
import pandas as pd
import numpy as np
import pickle
import json
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import sklearn.utils

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

parser = argparse.ArgumentParser(description='agglomerative cluster')

parser.add_argument('--operMode', type=str, default='TRAINING',
                    help='The mode component work on')

parser.add_argument('--load_model', type=str, default='./rf.pkl',
                    help='Pretrain model path')

parser.add_argument('--has_label', type=bool, default=True,
                    help='Data label  or not, evaluation or not')

parser.add_argument('--label_name', type=str, default='species',
                    help='Label col name')

parser.add_argument('--split_ratio', type=float, default=0.8,
                    help="Ratio between train data and validation data")

parser.add_argument('--shuffle', type=bool, default=True,
                    help='Shuffle dataset and split to train and validation')

parser.add_argument('--n_estimators', type=int, default=10,
                    help='Number of decision trees in forest')

parser.add_argument('--criterion', type=str, default="gini",
                    help="Measrue the quality of a split")

parser.add_argument('--max_features', type=float, default=None,
                    help='Ratio, max featrues to consider when looking for the best split')

parser.add_argument('--max_depth', type=int, default=2,
                    help='The maximun depth of the trees')

parser.add_argument('--min_samples_split', type=float, default=0.5,
                    help='Ratio, minimun number of samples required to split an internal node')

parser.add_argument('--min_samples_leaf', type=float, default=0.25,
                    help='Ratio, minimum number of smaples required to be at a leaf node')

parser.add_argument('--min_weight_fraction_leaf', type=float, default=0.0,
                    help='Ratio, minimum weighted fraction of the sum total of wieights required to be at a leaf node')

parser.add_argument('--max_leaf_nodes', type=int, default=None,
                    help='Grow trees with max_leaf_nodes in best-first fashion')

parser.add_argument('--min_impurity_decrease', type=float, default=0.0,
                    help='a node will be split if this shplit induces a decrease of the impurity greater than or euqal '
                         'to this vlaue')

parser.add_argument('--bootstrap', type=bool, default=True,
                    help='Whether bootstrap samples are used when building trees')

parser.add_argument('--n_jobs', type=int, default=1,
                    help='The number of jobs to run in parallel')

parser.add_argument('--class_weight_mode', type=str, default=None,
                    help='Mode class weight')
# class_weight可以设置成参数组，而且只有当mode='自定义'才显示

args = parser.parse_args()


def train_val_split(df, ratio=0.8, shuffle=False):
    if shuffle:
        df = sklearn.utils.shuffle(df)
    num_train = int(len(df) * ratio)
    tra_df = df[0:num_train]
    val_df = df[num_train:]
    return tra_df, val_df


def top_k(y_true, y_pred, label_name, k=1):
    nrows = y_pred.shape[0]
    if k > y_pred.shape[1]:
        return None

    pred_correct = np.zeros([nrows], dtype=np.int32)
    y_true_label = [label_name.tolist().index(i) for i in y_true]

    for i in range(nrows):
        target = y_true_label[i]
        pred_score = y_pred[i]
        top_k_idx = pred_score.argsort()[-k:][::-1]
        if target in top_k_idx:
            pred_correct[i] = 1
        else:
            pred_correct[i] = 0
    return np.mean(pred_correct)


def main():
    operMode = args.operMode
    logging.info('Random fortest work on operMode: {}'.format(operMode))

    input_in1_file = 'iris.csv'
    df = pd.read_csv(input_in1_file)
    if operMode == 'TRAINING':
        label_name = args.label_name
        n_estimators = args.n_estimators
        shuffle = args.shuffle
        split_ratio = args.split_ratio
        criterion = args.criterion
        max_features = args.max_features
        max_depth = args.max_depth
        min_samples_split = args.min_samples_split
        min_samples_leaf = args.min_samples_leaf
        min_weight_fraction_leaf = args.min_weight_fraction_leaf
        max_leaf_nodes = args.max_leaf_nodes
        min_impurity_decrease = args.min_impurity_decrease
        bootstrap = args.bootstrap
        n_jobs = args.n_jobs

        logging.info('model parameter as follow:\n'
                     'label_name: {}\n'
                     'n_estimators: {}\n'
                     'split_ratio: {}\n'
                     'shuffle: {}\n'
                     'criterion: {}\n'
                     'max_featrues: {}\n'
                     'max_depth: {}\n'
                     'min_samples_split: {}\n'
                     'min_samples_leaf: {}\n'
                     'min_weight_fraction_leaf: {}\n'
                     'max_leaf_nodes: {}\n'
                     'min_impurity_decrease: {}\n'
                     'bootstrap: {}\n'
                     'n_jobs: {}'.format(label_name, n_estimators, split_ratio, shuffle, criterion,
                                         max_features, max_depth, min_samples_split,
                                         min_samples_leaf, min_weight_fraction_leaf,
                                         max_leaf_nodes, min_impurity_decrease, bootstrap, n_jobs))

        tra_df, val_df = train_val_split(df, ratio=split_ratio, shuffle=shuffle)
        columns = df.columns.tolist()
        tra_y = tra_df[label_name].values
        val_y = val_df[label_name].values
        columns.remove(label_name)
        tra_x = tra_df[columns].values
        val_x = val_df[columns].values

        logging.info("Random Fortest Training Start...")
        try:
            clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                         max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                         bootstrap=bootstrap, n_jobs=n_jobs).fit(tra_x, tra_y)
        except Exception as e:
            logging.error("Unexpected Error {}".format(e))
            exit(0)

        logging.info("Random Fortest Training End and Stroe Model...")
        with open("rf.pkl", "wb") as f:
            pickle.dump(clf, f)

        val_y_pred_prob = clf.predict_proba(val_x)
        val_y_pred_label = clf.predict(val_x)

        cfmt = confusion_matrix(val_y, val_y_pred_label).tolist()

        top1_acc = top_k(val_y, val_y_pred_prob, clf.classes_, k=1)
        top5_acc = top_k(val_y, val_y_pred_prob, clf.classes_, k=5)
        fprs = []
        tprs = []
        aucs = []
        recalls = []
        precisions = []
        aps = []

        for c in range(len(clf.classes_)):
            val_y_true_binary = val_y == clf.classes_[c]
            val_y_pred_binary = val_y_pred_prob[:, c]
            fpr, tpr, thres_roc = roc_curve(val_y_true_binary, val_y_pred_binary, pos_label=1)
            auc = roc_auc_score(val_y_true_binary, val_y_pred_binary)
            precision, recall, thres_pr = precision_recall_curve(val_y_true_binary, val_y_pred_binary)
            ap = average_precision_score(val_y_true_binary, val_y_pred_binary)
            fprs.append(fpr.tolist())
            tprs.append(tpr.tolist())
            aucs.append(auc)
            recalls.append(recall.tolist())
            precisions.append(precision.tolist())
            aps.append(ap)

        pfmn_dict = {}
        pfmn_dict['graphs'] = []
        # ROC曲线
        graph_roc = {}
        graph_roc['name'] = 'ROC曲线'
        graph_roc['x_title'] = 'fpr'
        graph_roc['y_title'] = 'tpr'
        graph_roc['lines'] = []
        for i in range(len(fprs)):
            line = {}
            line['name'] = 'label为{}的ROC曲线'.format(i)
            line['relative'] = []
            relative = {}
            relative['name'] = 'auc'
            relative['value'] = aucs[i]
            line['relative'].append(relative)
            line['x_axis'] = fprs[i]
            line['y_axis'] = tprs[i]
            graph_roc['lines'].append(line)
        pfmn_dict['graphs'].append(graph_roc)
        # PR曲线
        graph_pr = {}
        graph_pr['name'] = 'PR曲线'
        graph_pr['x_title'] = 'recall',
        graph_pr['y_title'] = 'precision'
        graph_pr['lines'] = []
        for i in range(len(recalls)):
            line = {}
            line['name'] = 'label为{}的PR曲线'.format(i)
            line['relative'] = []
            relative = {}
            relative['name'] = 'ap'
            relative['value'] = aps[i]
            line['relative'].append(relative)
            line['x_axis'] = recalls[i]
            line['y_axis'] = precisions[i]
            graph_pr['lines'].append(line)
        pfmn_dict['graphs'].append(graph_pr)

        # 混淆矩阵
        pfmn_dict['matrixs'] = []
        matrix = {}
        matrix['name'] = '混淆矩阵'
        matrix['col_name'] = clf.classes_.tolist()
        matrix['row_name'] = clf.classes_.tolist()
        matrix['elements'] = cfmt
        pfmn_dict['matrixs'].append(matrix)
        # 数值型指标
        pfmn_dict['evaluation'] = []
        evals_top1 = {}
        evals_top1['name'] = "top1"
        evals_top1['value'] = top1_acc
        pfmn_dict['evaluation'].append(evals_top1)
        if top5_acc:
            evals_top5 = {}
            evals_top5['name'] = 'top5'
            evals_top5['value'] = top5_acc
            pfmn['evaluation'].append(evals_top5)

        pfmn_str = json.dumps(pfmn_dict)
        with open('pfmn.json', 'w') as f:
            f.write(pfmn_str)
        logging.info('Random Fortest Model Evaluation finished!')
    elif operMode == 'PREDICTION':
        has_label = args.has_label
        label_name = args.label_name
        load_model = args.load_model

        logging.info('model parameter configure as follow:\n'
                     'has_label: {}\n'
                     'label_name: {}\n'
                     'load_model: {}\n'.format(has_label, label_name, load_model))
        if has_label:
            if label_name is None:
                try:
                    raise Exception('if parameter has_label is true, label_name must not be none')
                except Exception as e:
                    logging.error(e)
                    exit(0)
        if has_label:
            columns = df.columns.tolist()
            test_y = df[label_name].values
            columns.remove(label_name)
            test_x = df[columns].values
        else:
            test_x = df.values

        logging.info("Random Fortest Load Model ")
        model_path = load_model
        if not os.path.exists(model_path):
            try:
                raise Exception('model file {} will be loaded not exists!'.format(model_path))
            except Exception as e:
                logging.error('Unexpected Error {}'.format(e))
                exit(0)
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        test_y_pred_prob = clf.predict_proba(test_x)
        if has_label:
            fprs = []
            tprs = []
            aucs = []
            recalls = []
            precisions = []
            aps = []
            for c in range(len(clf.classes_)):
                test_y_true_binary = test_y == clf.classes_[c]
                test_y_pred_binary = test_y_pred_prob[:, c]
                fpr, tpr, thres_roc = roc_curve(test_y_true_binary, test_y_pred_binary, pos_label=1)
                auc = roc_auc_score(test_y_true_binary, test_y_pred_binary)
                precision, recall, thres_pr = precision_recall_curve(test_y_true_binary, test_y_pred_binary)
                ap = average_precision_score(test_y_true_binary, test_y_pred_binary)
                fprs.append(fpr.tolist())
                tprs.append(tpr.tolist())
                aucs.append(auc)
                recalls.append(recall.tolist())
                precisions.append(precision.tolist())
                aps.append(ap)
            test_y_pred_label = clf.predict(test_x)
            cfmt = confusion_matrix(test_y, test_y_pred_label).tolist()
            top1_acc = top_k(test_y, test_y_pred_prob, clf.classes_, k=1)
            top5_acc = top_k(test_y, test_y_pred_prob, clf.classes_, k=5)
            pfmn_dict = {}
            pfmn_dict['graphs'] = []
            # ROC曲线
            graph_roc = {}
            graph_roc['name'] = 'ROC曲线'
            graph_roc['x_title'] = 'fpr'
            graph_roc['y_title'] = 'tpr'
            graph_roc['lines'] = []
            for i in range(len(fprs)):
                line = {}
                line['name'] = 'label为{}的ROC曲线'.format(i)
                line['relative'] = []
                relative = {}
                relative['name'] = 'auc'
                relative['value'] = aucs[i]
                line['relative'].append(relative)
                line['x_axis'] = fprs[i]
                line['y_axis'] = tprs[i]
                graph_roc['lines'].append(line)
            pfmn_dict['graphs'].append(graph_roc)
            # PR曲线
            graph_pr = {}
            graph_pr['name'] = 'PR曲线'
            graph_pr['x_title'] = 'recall',
            graph_pr['y_title'] = 'precision'
            graph_pr['lines'] = []
            for i in range(len(recalls)):
                line = {}
                line['name'] = 'label为{}的PR曲线'.format(i)
                line['relative'] = []
                relative = {}
                relative['name'] = 'ap'
                relative['value'] = aps[i]
                line['relative'].append(relative)
                line['x_axis'] = recalls[i]
                line['y_axis'] = precisions[i]
                graph_pr['lines'].append(line)
            pfmn_dict['graphs'].append(graph_pr)

            # 混淆矩阵
            pfmn_dict['matrixs'] = []
            matrix = {}
            matrix['name'] = '混淆矩阵'
            matrix['col_name'] = clf.classes_.tolist()
            matrix['row_name'] = clf.classes_.tolist()
            matrix['elements'] = cfmt
            pfmn_dict['matrixs'].append(matrix)
            # 数值型指标
            pfmn_dict['evaluation'] = []
            evals_top1 = {}
            evals_top1['name'] = "top1"
            evals_top1['value'] = top1_acc
            pfmn_dict['evaluation'].append(evals_top1)
            if top5_acc:
                evals_top5 = {}
                evals_top5['name'] = 'top5'
                evals_top5['value'] = top5_acc
                pfmn_dict['evaluation'].append(evals_top5)

            pfmn_str = json.dumps(pfmn_dict)
            with open('pfmn.json', 'w') as f:
                f.write(pfmn_str)
    else:
        logging.fatal('Random fortest not support {}'.format(operMode))
        raise Exception('Random fortest not support {}'.format(operMode))


if __name__ == '__main__':
    main()
