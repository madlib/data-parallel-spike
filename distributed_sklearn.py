#!/usr/bin/env python -W ignore::DeprecationWarning
from __future__ import print_function
import itertools

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.base import clone
import matplotlib.pyplot as plt

import numpy as np
import math
import copy

import cPickle
import gzip

import csv
USE_IRIS = False
N_ITERATIONS = 50
ACC_THRESHOLD = 0.85


# code from scikit-learn
# replace self with model
def _init_coef(self, fan_in, fan_out):
    if self.activation == 'logistic':
        # Use the initialization method recommended by
        # Glorot et al.
        init_bound = np.sqrt(2. / (fan_in + fan_out))
    elif self.activation in ('identity', 'tanh', 'relu'):
        init_bound = np.sqrt(6. / (fan_in + fan_out))
    else:
        # this was caught earlier, just to make sure
        raise ValueError("Unknown activation function %s" %
                         self.activation)

    coef_init = self._random_state.uniform(-init_bound, init_bound,
                                           (fan_in, fan_out))
    intercept_init = self._random_state.uniform(-init_bound, init_bound,
                                                fan_out)
    return coef_init, intercept_init


class MLPClassifierWithInitialize(MLPClassifier):
    def _init_coef(self, fan_in, fan_out):
        if self.activation == 'logistic':
            init_bound = np.sqrt(2. / (fan_in + fan_out))
        elif self.activation in ('identity', 'tanh', 'relu'):
            init_bound = np.sqrt(6. / (fan_in + fan_out))
        else:
            raise ValueError("Unknown activation function %s" %
                             self.activation)

        coef_init = self._random_state.uniform(-init_bound, init_bound,
                                               (fan_in, fan_out))
        intercept_init = self._random_state.uniform(-init_bound, init_bound,
                                                    fan_out)
        print(coef_init, intercept_init)
        return coef_init, intercept_init
# end of code from scikit-learn


def fetch_mnist():
    # Load the dataset
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_test_split(
        np.concatenate((train_set[0], valid_set[0], test_set[0])) / 255.,
        np.concatenate((train_set[1], valid_set[1], test_set[1])),
        train_size=60000)


class distributed_training:
    def __init__(self, n_segments, run_centralized=False):
        self.n_classes = 3
        self.n_iterations = N_ITERATIONS
        self.batch_size = 25
        self.n_segments = n_segments
        self.n_epochs = 1
        self.get_data()
        self.distribute_data()
        self.aggregate_model = self.define_segment_models()
        self.train_model_aggregate(run_centralized)

    def get_data(self):
        if USE_IRIS:
            iris = datasets.load_iris()
            self.x_train, self.y_train = iris.data, iris.target
            print('Class distribution in whole dataset:', np.bincount(self.y_train.astype('int64')))
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = fetch_mnist()
            print('Class distribution in whole dataset:', np.bincount(self.y_train.astype('int64')))

    def distribute_data(self):
        self.segment_batches = []
        self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        data_per_segment = int(math.floor(self.x_train.shape[0] / self.n_segments))
        for i in range(self.n_segments):
            self.segment_batches.append((self.x_train[data_per_segment * i:data_per_segment * i + data_per_segment],
                                         self.y_train[data_per_segment * i:data_per_segment * i + data_per_segment]))
        print([(s[0].shape, s[1].shape) for s in self.segment_batches])

    def get_new_model(self):
        iris_model = MLPClassifier(
            hidden_layer_sizes=[3, ],
            # activation='relu',
            solver='sgd',
            learning_rate='constant',
            max_iter=self.n_epochs,
            learning_rate_init=0.05,
            momentum=0,
            batch_size=self.batch_size,
            random_state=0)

        mnist_model = MLPClassifier(
            hidden_layer_sizes=[10, ],
            # activation='relu',
            solver='adam',
            # learning_rate='invscaling',
            learning_rate='constant',
            max_iter=self.n_epochs,
            learning_rate_init=0.01,
            momentum=0.95,
            nesterovs_momentum=True,
            batch_size=self.batch_size,
            random_state=42,
            verbose=False,
            tol=1e-10,
            early_stopping=False)
        return (mnist_model, iris_model)[USE_IRIS]

    def define_segment_models(self):
        self.segment_models = []
        common_model = self.get_new_model()
        for i in range(self.n_segments):
            self.segment_models.append(clone(common_model))
        return common_model

    def aggregate_models(self):
        try:
            self.aggregate_model.coefs_ = list(sum([np.array(s.coefs_)
                                                    for s in self.segment_models]) / self.n_segments)
            self.aggregate_model.intercepts_ = list(sum([np.array(s.intercepts_)
                                                         for s in self.segment_models]) / self.n_segments)
        except Exception as e:
            print(str(e))
            print([np.array(s.coefs_) for s in self.segment_models])
            print([np.array(s.intercepts_) for s in self.segment_models])
            raise

        self.aggregate_model.classes_ = self.segment_models[0].classes_[:]
        self.aggregate_model._label_binarizer = self.segment_models[0]._label_binarizer
        self.aggregate_model.n_layers_ = self.segment_models[0].n_layers_
        self.aggregate_model.n_outputs_ = self.segment_models[0].n_outputs_
        self.aggregate_model.out_activation_ = self.segment_models[0].out_activation_

    def reset(self, model):
        model.n_iter_ = 0
        model.t_ = 0
        model.loss_curve_ = []
        model._no_improvement_count = 0
        model.best_loss_ = np.inf

    def clone_models(self):
        for i in range(self.n_segments):
            self.segment_models[i] = clone(self.aggregate_model)
            self.reset(self.segment_models[i])
            self.segment_models[i].coefs_ = copy.deepcopy(self.aggregate_model.coefs_)
            self.segment_models[i].intercepts_ = copy.deepcopy(self.aggregate_model.intercepts_)

            self.segment_models[i].classes_ = self.aggregate_model.classes_[:]
            self.segment_models[i].n_layers_ = self.aggregate_model.n_layers_
            self.segment_models[i].n_outputs_ = self.aggregate_model.n_outputs_
            self.segment_models[i]._label_binarizer = self.aggregate_model._label_binarizer
            self.segment_models[i].out_activation_ = self.aggregate_model.out_activation_

    def train_model_aggregate(self, run_centralized=False):
        self.agg_model_scores = []
        self.segment_models_scores = []
        self.centralized_model_scores = []

        # Training and evaluation for a single model with all training data
        if run_centralized or self.n_segments == 1:
            self.centralized_model = clone(self.aggregate_model)
            self.centralized_model.warm_start = True
            self.centralized_model.fit(self.x_train, self.y_train)
            for i in range(self.n_iterations):
                centralized_score = self.centralized_model.score(self.x_train, self.y_train)
                self.centralized_model_scores.append(centralized_score)
                print("Iteration = {}, Centralized score = {}".format(i+1, centralized_score))
                self.centralized_model.partial_fit(self.x_train, self.y_train)
        else:
            print('------------------- Nu of segments = {} ----------------------'.
                  format(self.n_segments))
            # Training and evaluation loop for aggregating models
            for i in range(self.n_iterations):
                # print("Iteration:", i + 1, "/", self.n_iterations, end=' - ')
                self.segment_models_scores.append(list(itertools.repeat(0, self.n_segments)))
                for seg_index, model_seg in enumerate(self.segment_models):
                    (x_train_seg, y_train_seg) = self.segment_batches[seg_index]
                    if i == 0:
                        model_seg.fit(x_train_seg, y_train_seg)
                    else:
                        model_seg.partial_fit(x_train_seg, y_train_seg)
                    model_score = model_seg.score(self.x_train, self.y_train)
                    self.segment_models_scores[i][seg_index] = model_score
                    # print("\t Segment model {} score = {}".format(seg_index, model_score))

                self.aggregate_models()
                self.clone_models()
                agg_score = self.aggregate_model.score(self.x_train, self.y_train)
                self.agg_model_scores.append(agg_score)
                print("Iteration {}/{}, aggregate model score = {}".
                      format(i + 1, self.n_iterations, agg_score))
            print('---------------------------------------------------------------')


def run_experiment():
    for n_segments in ([1] + range(2, 1, 2)):   # range(start, stop, step)
        model = distributed_training(n_segments, run_centralized=True)
        if hasattr(model, 'centralized_model_scores'):
            with open('mnist_centralized_scores.csv', 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for val in model.centralized_model_scores:
                    csvwriter.writerow([val])
        with open('mnist_distributed_scores_seg_{}.csv'.format(n_segments), 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                   quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for seg_scores, agg_score in zip(model.segment_models_scores, model.agg_model_scores):
                csvwriter.writerow(list(seg_scores) + [agg_score])


def compute_cutoff():
    all_cutoffs = []
    segment_counts = [1] + range(2, 11, 2)
    for n_segments in segment_counts:
        if n_segments == 1:
            filename = 'mnist_centralized_scores.csv'
        else:
            filename = 'mnist_distributed_scores_seg_{}.csv'.format(n_segments)
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for n_iter, line in enumerate(csvreader):
                if float(line[-1]) >= ACC_THRESHOLD:
                    all_cutoffs.append(n_iter + 1)
                    break
            else:
                all_cutoffs.append(N_ITERATIONS)

    # plot comparison between single-node and multi-node convergencee
    fig = plt.figure()
    fig.suptitle('MNIST data classification', fontweight='bold')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Time to reach {} accuracy'.format(ACC_THRESHOLD))
    ax.set_xlabel('Number of segments')
    ax.set_ylabel('Number of iterations')
    linear_scale_line = np.array(segment_counts) * all_cutoffs[0]
    plt.plot(segment_counts, all_cutoffs, 'g-', marker='+', label="Actual convergence times")
    plt.plot(segment_counts, linear_scale_line, 'b-', alpha=0.5, label="Break-even wrt to centralized")
    ax.legend(loc='lower right')
    plt.savefig('mnist_accuracy_cutoffs.png')


def plot_model_scores():
    data_label = 'MNIST'
    # data_label = 'IRIS'
    suptitle = '{} data classification'.format(data_label)

    centralized_model_scores = []
    with open('mnist_centralized_scores.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for line in csvreader:
            centralized_model_scores.append(float(line[0]))
    x_axis = range(N_ITERATIONS)

    all_agg_models = []
    segment_counts = range(2, 11, 2)
    for n_segments in segment_counts:   # range(start, stop, step)
        fig = plt.figure()
        fig.suptitle(suptitle, fontweight='bold')
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)
        ax.set_title('Number of segments = {}'.format(n_segments))
        ax.set_xlabel('Number of iterations')
        ax.set_ylabel('Accuracy')
        plt.ylim((0, 1))

        segment_models_scores = []
        agg_model_scores = []
        with open('mnist_distributed_scores_seg_{}.csv'.format(n_segments), 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for line in csvreader:
                segment_models_scores.append([float(i) for i in line[:-1]])
                agg_model_scores.append(float(line[-1]))
        print(len(segment_models_scores), len(agg_model_scores))
        segment_models_scores = np.transpose(np.array(segment_models_scores))
        all_agg_models.append(agg_model_scores)

        for scores in segment_models_scores:
            plt.plot(x_axis, scores, 'k-', alpha=0.2)
        plt.plot(x_axis, agg_model_scores, 'b-', label='Aggregated model')
        plt.plot(x_axis, centralized_model_scores, 'g-', label='Centralized model')
        ax.legend(loc='lower right')
        plt.savefig('{}_scores_seg_{}.png'.format(data_label.lower(), n_segments))

    # plot comparison between single-node and multi-node convergencee
    fig = plt.figure()
    fig.suptitle(suptitle, fontweight='bold')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Centralized vs Distributed'.format(n_segments))
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Accuracy')
    plt.ylim((0, 1))
    plt.plot(x_axis, centralized_model_scores, 'g-', label='Centralized model')
    for n_segments, each_agg_model in zip(segment_counts, all_agg_models):
        plt.plot(x_axis, each_agg_model,
                 label='{} segments'.format(n_segments))
    ax.legend(loc='lower right')
    plt.savefig('{}_scores_comparison.png'.format(data_label.lower()))


import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    run_experiment()
    plot_model_scores()
    compute_cutoff()
