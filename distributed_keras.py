#!/usr/bin/env python -W ignore::DeprecationWarning
from __future__ import print_function
import itertools

import keras
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np
import math

import csv
USE_CIFAR = False
N_ITERATIONS = 15
ACC_THRESHOLD = 0.6


class distributed_training:
    def __init__(self, n_segments, run_centralized=False):
        self.n_classes = 10
        self.n_iterations = N_ITERATIONS
        self.batch_size = 32
        self.n_segments = n_segments
        self.n_epochs = 1
        self.get_data()
        self.distribute_data()
        self.aggregate_model = self.define_segment_models()
        self.train_model_aggregate(run_centralized)

    def get_data(self):
        if USE_CIFAR:
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            self.num_train_examples, self.num_test_examples = x_train.shape[0], x_test.shape[0]
            self.img_rows, self.img_cols, self.num_channels = x_train.shape[2], x_train.shape[3], x_train.shape[1]
            x_train = x_train.reshape(self.num_train_examples, self.num_channels, self.img_rows, self.img_cols)
            x_test = x_test.reshape(self.num_test_examples, self.num_channels, self.img_rows, self.img_cols)
        else:
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            self.num_train_examples, self.num_test_examples = x_train.shape[0], x_test.shape[0]
            self.img_rows, self.img_cols, self.num_channels = x_train.shape[1], x_train.shape[2], 1
            x_train = x_train.reshape(self.num_train_examples, self.img_rows, self.img_cols, self.num_channels)
            x_test = x_test.reshape(self.num_test_examples, self.img_rows, self.img_cols, self.num_channels)

        self.y_train = y_train.reshape(self.num_train_examples, 1)
        self.y_test = y_test.reshape(self.num_test_examples, 1)
        self.x_train = x_train.astype('float32')
        self.x_test = x_test.astype('float32')
        self.x_train /= 255.0
        self.x_test /= 255.0
        # Convert the y vectors to categorical format for crossentropy prediction
        self.y_train = keras.utils.to_categorical(self.y_train, self.n_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.n_classes)

    def distribute_data(self):
        self.segment_batches = []
        # self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        data_per_segment = int(math.floor(self.x_train.shape[0] / self.n_segments))
        for i in range(self.n_segments):
            self.segment_batches.append((self.x_train[data_per_segment * i:data_per_segment * i + data_per_segment],
                                         self.y_train[data_per_segment * i:data_per_segment * i + data_per_segment]))
        print([(s[0].shape, s[1].shape) for s in self.segment_batches])

    def get_new_model(self):
        model = Sequential()
        if USE_CIFAR:
            model.add(Conv2D(32, kernel_size=(3, 3),
                             activation='relu',
                             input_shape=(self.num_channels, self.img_rows, self.img_cols,)))
        else:
            model.add(Conv2D(3, kernel_size=(3, 3),
                             activation='sigmoid',
                             input_shape=(self.img_rows, self.img_cols, self.num_channels,)))
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['categorical_accuracy'])
        return model

    def define_segment_models(self):
        self.segment_models = []
        common_model = self.get_new_model()
        for i in range(self.n_segments):
            cloned_model = self.get_new_model()
            cloned_model.set_weights(common_model.get_weights())
            cloned_model.compile(loss='categorical_crossentropy',
                                 optimizer=Adam(),
                                 metrics=['categorical_accuracy'])
            self.segment_models.append(cloned_model)
        return common_model

    def aggregate_from_segments(self):
        # Compile aggregate model
        # self.aggregate_model.compile(loss='categorical_crossentropy',
        #                              optimizer=Adam(),
        #                              metrics=['accuracy'])
        try:
            avg_weights = sum(
                [np.array(s.get_weights()) for s in self.segment_models]) / self.n_segments
            self.aggregate_model.set_weights(avg_weights)
        except Exception as e:
            print(str(e))
            for s in self.segment_models:
                print('------------------------------------------------------')
                print([i.shape for i in s.get_weights()])
                print('------------------------------------------------------')
            raise

    def clone_to_segments(self):
        for model in self.segment_models:
            model.set_weights(self.aggregate_model.get_weights())

    def train_model_aggregate(self, run_centralized=False):
        initial_model_score = self.aggregate_model.evaluate(self.x_train, self.y_train, verbose=0)[1]
        self.agg_model_scores = [initial_model_score]
        self.segment_models_scores = [[initial_model_score] * self.n_segments]

        # Training and evaluation for a single model with all training data
        if run_centralized:
            self.centralized_model_scores = [initial_model_score]
            self.centralized_model = self.get_new_model()
            self.centralized_model.set_weights(self.aggregate_model.get_weights())
            # self.centralized_model.warm_start = True
            self.centralized_model.compile(loss='categorical_crossentropy',
                                           optimizer=Adam(),
                                           metrics=['categorical_accuracy'])
            for i in range(self.n_iterations):
                self.centralized_model.fit(self.x_train, self.y_train,
                                           batch_size=self.batch_size,
                                           epochs=1,
                                           verbose=0)
                centralized_score = self.centralized_model.evaluate(self.x_train,
                                                                    self.y_train,
                                                                    verbose=0)[1]
                print("Iteration {}: centralized score = {}".format(i, centralized_score))
                self.centralized_model_scores.append(centralized_score)

        print('------------------- Nu of segments = {} ----------------------'.
              format(self.n_segments))

        # Training and evaluation loop for aggregating models
        # for i in itertools.count():
        # for i in range(self.n_iterations):
        if False:
            self.segment_models_scores.append(list(itertools.repeat(0, self.n_segments)))
            for seg_index, model_seg in enumerate(self.segment_models):
                (x_train_seg, y_train_seg) = self.segment_batches[seg_index]
                model_seg.fit(x_train_seg, y_train_seg,
                              batch_size=self.batch_size,
                              verbose=0, epochs=1)

                model_score = model_seg.evaluate(self.x_train, self.y_train, verbose=0)[1]
                print("\t Segment model {} score = {}".format(seg_index, model_score))
                self.segment_models_scores[i+1][seg_index] = model_score

            self.aggregate_from_segments()
            self.clone_to_segments()
            agg_score = self.aggregate_model.evaluate(self.x_train, self.y_train, verbose=0)[1]
            self.agg_model_scores.append(agg_score)
            print("Iteration {}/{}, aggregate model score = {}".
                  format(i+1, self.n_iterations, agg_score))
            # if agg_score >= ACC_THRESHOLD or (i+1) == self.n_iterations:
            #     break
        print('---------------------------------------------------------------')


def run_experiment():
    for n_segments in range(2, 3, 2):   # range(start, stop, step)
        model = distributed_training(n_segments, run_centralized=True)
        if hasattr(model, 'centralized_model_scores'):
            with open('mnist_keras_scores_seg_1.csv', 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for val in model.centralized_model_scores:
                    csvwriter.writerow([val])
        with open('mnist_keras_scores_seg_{}.csv'.format(n_segments), 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                   quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for seg_scores, agg_score in zip(model.segment_models_scores, model.agg_model_scores):
                csvwriter.writerow(seg_scores + [agg_score])


def compute_cutoff():
    data_label = 'CIFAR'  # 'MNIST'
    suptitle = '{} data classification'.format(data_label)
    all_cutoffs = []
    segment_counts = [1] + range(2, 11, 2)
    for n_segments in segment_counts:
        filename = '{}_keras_scores_seg_{}.csv'.format(data_label.lower(), n_segments)
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
    fig.suptitle(suptitle, fontweight='bold')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Time to reach {} accuracy'.format(ACC_THRESHOLD))
    ax.set_xlabel('Number of segments')
    ax.set_ylabel('Number of iterations')
    linear_scale_line = np.array(segment_counts) * all_cutoffs[0]
    plt.plot(segment_counts, all_cutoffs, 'g-', marker='+', label="Actual convergence times")
    plt.plot(segment_counts, linear_scale_line, 'b-', alpha=0.5, label="Break-even wrt to centralized")
    ax.legend(loc='lower right')
    plt.savefig('{}_keras_accuracy_cutoffs.png'.format(data_label.lower()))


def plot_model_scores():
    for data_label in ('CIFAR', 'MNIST'):
        suptitle = '{} data classification'.format(data_label)

        x_axis = range(N_ITERATIONS + 1)
        segment_counts = [1] + range(2, 11, 2)   # first one should always be 1
        all_single_models = []
        for n_segments in segment_counts:
            segment_models_scores = []
            single_model_scores = []
            with open('{}_keras_scores_seg_{}.csv'.format(data_label.lower(), n_segments), 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for line in csvreader:
                    segment_models_scores.append([float(i) for i in line[:-1]])
                    if n_segments > 1:
                        single_model_scores.append(float(line[-1]))
                    else:
                        single_model_scores.append(float(line[0]))
            all_single_models.append(single_model_scores)
            if n_segments > 1:
                fig = plt.figure()
                fig.suptitle(suptitle, fontweight='bold')
                ax = fig.add_subplot(111)
                fig.subplots_adjust(top=0.85)
                ax.set_title('Number of segments = {}'.format(n_segments))
                ax.set_xlabel('Number of iterations')
                ax.set_ylabel('Accuracy')
                plt.ylim((0, 1))
                segment_models_scores = np.transpose(np.array(segment_models_scores))
                for scores in segment_models_scores[1:]:
                    plt.plot(x_axis, scores, 'k-', alpha=0.2)
                plt.plot(x_axis, single_model_scores, 'b-', label='Aggregated model')
                plt.plot(x_axis, all_single_models[0], 'g-', label='Centralized model')
                ax.legend(loc='lower right')
                plt.savefig('{}_keras_scores_seg_{}.png'.format(data_label.lower(), n_segments))

        # plot comparison between single-node and multi-node convergencee
        fig = plt.figure()
        fig.suptitle(suptitle, fontweight='bold')
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)
        ax.set_title('Centralized vs Distributed')
        ax.set_xlabel('Number of iterations')
        ax.set_ylabel('Accuracy')
        plt.ylim((0, 1))
        for n_segments, each_agg_model in zip(segment_counts, all_single_models):
            if n_segments > 1:
                plt.plot(x_axis, each_agg_model, label='{} segments'.format(n_segments))
            else:
                plt.plot(x_axis, each_agg_model, 'g-', marker='+', label='Centralized model')
        ax.legend(loc='lower right')
        plt.savefig('{}_keras_scores_comparison.png'.format(data_label.lower()))


import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # run_experiment()
    # plot_model_scores()
    compute_cutoff()
