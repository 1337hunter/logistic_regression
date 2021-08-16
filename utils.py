import csv
import numpy as np
from logistic_regression import LogisticRegression
import ast
import copy
import math

def LoadDataset(path, features, target=None):
    it = csv.DictReader(path)
    X = []
    y = []
    with open(path, 'r') as file:
        it = csv.DictReader(file)
        for row in it:
            X += [[np.nan if row[x] == '' else float(row[x]) for x in features]]
            if target is not None:
                y += [row[target]]
    return X, y


def DropNan(X, y=None):
    i = 0
    while i < len(X):
        if np.nan in X[i] or (y is not None and y[i] == ''):
            del X[i]
            if y is not None:
                del y[i]
        else:
            i += 1

def FillNan(X, mean):
    i = 0
    while i < X.shape[0]:
        j = 0
        while j < len(X[i]):
            if np.isnan(X[i][j]):
                X[i][j] = mean[j]
            j += 1
        i += 1

def find_nearest_naighbor(j, row, X):
    row[j] = 0
    i = 0
    min_dist = math.inf
    min_index = -1
    while i < X.shape[0]:
        has_nan = False
        j = 0
        while j < len(X[i]):
            if np.isnan(X[i][j]):
                has_nan = True
                break
            j += 1
        if not has_nan:
            dist = np.sqrt(((X[i] - row) * (X[i] - row)).sum())
            if dist < min_dist:
                min_dist = dist
                min_index = i
        i += 1
    return X[min_index][j - 1]





def FillNanKNN(X, X_train):
    i = 0
    while i < X.shape[0]:
        j = 0
        while j < len(X[i]):
            if np.isnan(X[i][j]):
                X[i][j] = find_nearest_naighbor(j, copy.copy(X[i]), X_train)
            j += 1
        i += 1

def save_predictions(path, pred):
    with open(path, 'w+') as file:
        i = 0
        file.write("Index,Hogwarts House\n")
        while i < len(pred):
            file.write(f'{str(i)},{pred[i]}\n')
            i += 1


def accuracy(pred, y):
    true_ = 0
    false_ = 0
    pred = np.array(pred)
    y = np.array(y)
    if pred.shape != y.shape:
        raise f"accuracy: shape error: {pred.shape}, {y.shape}"
    i = 0
    while i < y.shape[0]:
        if pred[i] == y[i]:
            true_ += 1
        else:
            false_ += 1
        i += 1
    return true_ / (true_ + false_)


def Normalize(X):
    X = np.array(X)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    return X, mean, std


def train_test_split(X, y, test_size=0.1, random_state=None):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    d = {}
    columns = []

    for x in y:
        if x not in columns:
            columns += [x]
    for col in columns:
        rez = np.where(y == col)[0]
        i = 0
        while i < rez.shape[0]:
            if i % (test_size * 100) == 0:
                X_test += [X[rez[i]]]
                y_test += [y[rez[i]]]
            else:
                X_train += [X[rez[i]]]
                y_train += [y[rez[i]]]
            i += 1
    np.random.seed(random_state)
    train_permutation = np.random.permutation(len(X_train))
    test_permutation = np.random.permutation(len(X_test))
    return  np.array(X_train)[train_permutation], \
            np.array(y_train)[train_permutation], \
            np.array(X_test)[test_permutation], \
            np.array(y_test)[test_permutation]


def dump(model, filename):
    with open(filename, 'w+') as file:
        file.write(str(model.__dict__))


def load(filename):
    model = LogisticRegression()
    with open(filename, 'r') as file:
        dict_ = ast.literal_eval(file.read().strip())
        model.__dict__ = dict_
    return model
