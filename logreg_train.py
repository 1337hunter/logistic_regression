#!/usr/bin/env python3

# Logistic regression one versus all

import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
from logistic_regression import LogisticRegression
from utils import *


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
    y = np.array(y)
    X = np.c_[[1] * X.shape[0], X]
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    pred = logreg.predict(X_test)
    print("Test accuracy:", accuracy(pred, y_test))
    dump(logreg, 'logreg.model')


if __name__ == '__main__':
    main()
