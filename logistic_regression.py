#!/usr/bin/env python3

# Logistic regression one versus all

import matplotlib.pyplot as plt
import csv
import numpy as np
import math
from tqdm import tqdm
import ast


class LogisticRegression:
    def __init__(self, optimizer=None, epoch=70, lr=7,
            random_state=None, batch_size=None, batch_shuffle=False):
        self.weights = {}
        self.epoch = epoch
        self.lr = lr
        self.optimizer = optimizer
        self.random_state = random_state
        self.batch_size = batch_size
        self.shuffle = batch_shuffle
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def h(self, X, weights, axis):
        return np.array(self.sigmoid((X * weights).sum(axis=axis)))

    def Loss(self, h, y):
        return -1 * np.array((y * np.log(h) + (1 - y) * np.log(1 - h))).mean()

    def fit_gradient_descent(self, X, y):
        loss = {}
        with tqdm(total=len(self.targets * self.epoch)) as tq:
            for target in self.targets:
                np.random.seed(1)
                self.weights[target] = (np.random.random(X.shape[1]))
                loss[target] = []
                code_y = np.array([1 if x == target else 0 for x in y])
                temp_weights = [0] * len(self.weights[target])
                lr = self.lr
                for i in range(self.epoch):
                    for j in range(len(temp_weights)):
                        temp = self.weights[target][j] - \
                                lr * ((self.h(X, self.weights[target], 1) - code_y) \
                                * X[:, j]).mean()
                        temp_weights[j] = temp.copy()
                    self.weights[target] = temp_weights.copy()
                    loss[target] += [self.Loss(self.h(X, self.weights[target], 1), code_y)]
                    tq.update(1)
        plt.show(block=False)
        for k, v in loss.items():
            plt.plot(list(range(self.epoch)), v, label=k)
        plt.title("Train loss")
        plt.legend(list(loss.keys()))
        plt.show()


    def fit_sgd(self, X, y):
        loss = {}
        with tqdm(total=len(self.targets * self.epoch)) as tq:
            for target in self.targets:
                np.random.seed(self.random_state)
                self.weights[target] = (np.random.random(X.shape[1]))
                loss[target] = []
                code_y = np.array([1 if x == target else 0 for x in y])
                temp_weights = [0] * len(self.weights[target])
                lr = self.lr
                train_index = len(y) - 1
                for i in range(self.epoch):
                    if train_index <= 0:
                        permutation = np.random.permutation(len(y))
                        X = X[permutation]
                        y = y[permutation]
                        train_index = len(y) - 1
                    else:
                        train_index -= 1
                    for j in range(len(temp_weights)):
                        temp = self.weights[target][j] - \
                            lr * (self.h(X[train_index], self.weights[target], 0) - \
                            code_y[train_index]) * X[train_index][j]
                        temp_weights[j] = temp.copy()
                    self.weights[target] = temp_weights.copy()
                    tq.update(1)
   
    def fit_bgd(self, X, y):
        loss = {}
        bs = self.batch_size
        with tqdm(total=len(self.targets * self.epoch)) as tq:
            for target in self.targets:
                np.random.seed(self.random_state)
                self.weights[target] = (np.random.random(X.shape[1]) * 2)
                loss[target] = []
                code_y = np.array([1 if x == target else 0 for x in y])
                temp_weights = [0] * len(self.weights[target])
                lr = self.lr
                for epoch in range(self.epoch):
                    for i in range(0, X.shape[0] - bs, bs):
                        for j in range(len(temp_weights)):
                            temp = self.weights[target][j] - \
                                    lr * ((self.h(X[i:i + bs], self.weights[target], 1) - \
                                    code_y[i:i + bs]) * X[i: i + bs, j]).mean()
                            temp_weights[j] = temp.copy()
                        self.weights[target] = temp_weights.copy()
                    if X.shape[0] % bs != 0:
                        start = X.shape[0] - X.shape[0] % bs
                        for j in range(len(temp_weights)):
                            temp = self.weights[target][j] - \
                                lr * ((self.h(X[start:], self.weights[target], 1) - \
                                code_y[start:]) * X[start:, j]).mean()
                            temp_weights[j] = temp.copy()
                        self.weights[target] = temp_weights.copy()
                    if self.shuffle:
                        permutation = np.random.permutation(len(y))
                        X = X[permutation]
                        y = y[permutation]
                    tq.update(1)


    def fit(self, X, y):
        self.targets = list(set(y))
        if self.optimizer is None:
            self.fit_gradient_descent(X, y)
        elif self.optimizer == "SGD":
            self.fit_sgd(X, y)
        elif self.optimizer == "BGD":
            self.fit_bgd(X, y)


    def predict(self, X):
        X = np.array(X)
        predictions = {}
        for t in self.targets:
            predictions[t] = self.sigmoid((X * self.weights[t]).sum(axis=1))
        i = 0
        results = []
        while i < X.shape[0]:
            max_ = -1
            target_ = ''
            for t in self.targets:
                if predictions[t][i] > max_:
                    target_ = t
                    max_ = predictions[t][i]
            results += [target_]
            i += 1
        return np.array(results)

    def unnormalize_weights(self, coef):
        for k in self.weights.keys():
            self.weights[k] = list(self.weights[k] * coef)

