#!/usr/bin/env python3

# Logistic regression one versus all

import matplotlib.pyplot as plt
import csv
import numpy as np
import math
from tqdm import tqdm

class LogisticRegression:
    def __init__(self, epoch=100, lr=1):
        self.weights = {}
        self.epoch = epoch
        self.lr = lr
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def h(self, X, weights):
        return np.array(self.sigmoid((X * weights).sum(axis=1)))

    def Loss(self, pred, y):
        return -1 * np.array((y * np.log(pred) + (1 - y) * np.log(1 - pred))).mean()

    def fit(self, X, y):
        self.targets = list(set(y))
        X = np.c_[[1] * X.shape[0], X]
        losses = {}
        for x in self.targets:
            self.weights[x] = np.array([0] * X.shape[1])
            losses[x] = []
        with tqdm(total=len(self.targets * self.epoch)) as tq:
            for target in self.targets:
                code_y = np.array([1 if x == target else 0 for x in y])
                temp_weights = [0] * len(self.weights[target])
                for i in range(self.epoch):
                    for j in range(len(temp_weights)):
                        temp = self.weights[target][j] - \
                                self.lr * ((self.h(X, self.weights[target]) - code_y) \
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





def LoadDataset(path, features, target):
    it = csv.DictReader(path)
    X = []
    y = []
    with open(path, 'r') as file:
        it = csv.DictReader(file)
        for row in it:
            X += [[np.nan if row[x] == '' else float(row[x]) for x in features]]
            y += [row[target]]
    return X, y

def DropNan(X, y):
    i = 0
    while i < len(X):
        if np.nan in X[i] or y[i] == '':
            del X[i]
            del y[i]
        else:
            i += 1

def Normalize(X):
    X = np.array(X)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    return X, mean, std



def main():
    path = './data/dataset_train.csv'
    features = ['Herbology', 'Astronomy', 'Ancient Runes']
    target = 'Hogwarts House'
    X, y = LoadDataset(path, features, target)
    if len(X) != len(y):
        print("X and y size does not match")
        exit()
    DropNan(X, y)
    X, mean, std = Normalize(X)
    logreg = LogisticRegression()
    logreg.fit(X, y)

if __name__ == '__main__':
    main()
