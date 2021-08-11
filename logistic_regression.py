#!/usr/bin/env python3

# Logistic regression one versus all

import matplotlib.pyplot as plt
import csv
import numpy as np
import math
from tqdm import tqdm
import ast


class LogisticRegression:
    def __init__(self, epoch=70, lr=7):
        self.weights = {}
        self.epoch = epoch
        self.lr = lr
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def h(self, X, weights):
        return np.array(self.sigmoid((X * weights).sum(axis=1)))

    def Loss(self, h, y):
        return -1 * np.array((y * np.log(h) + (1 - y) * np.log(1 - h))).mean()

    def fit(self, X, y):
        self.targets = list(set(y))
        #X = np.c_[[1] * X.shape[0], X]
        losses = {}
        for x in self.targets:
            #self.weights[x] = np.array([.0] * X.shape[1])
            np.random.seed(1)
            self.weights[x] = (np.random.random(X.shape[1]))
            print(self.weights[x])
            losses[x] = []
        with tqdm(total=len(self.targets * self.epoch)) as tq:
            for target in self.targets:
                code_y = np.array([1 if x == target else 0 for x in y])
                temp_weights = [0] * len(self.weights[target])
                lr = self.lr
                for i in range(self.epoch):
                    for j in range(len(temp_weights)):
                        temp = self.weights[target][j] - \
                                lr * ((self.h(X, self.weights[target]) - code_y) \
                                * X[:, j]).mean()
                        temp_weights[j] = temp.copy()
                    self.weights[target] = temp_weights.copy()
                    losses[target] += [self.Loss(self.h(X, self.weights[target]), code_y)]
                    tq.update(1)
        plt.show(block=False)
        for k, v in losses.items():
            plt.plot(list(range(self.epoch)), v, label=k)
        plt.title("Train loss")
        plt.legend(list(losses.keys()))
        plt.show()

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

