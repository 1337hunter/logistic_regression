#!/usr/bin/env python3

# Logistic regression one versus all

import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import logistic_regression
from utils import *
import sys

def main():
    path = './data/dataset_train.csv'
    features = ['Herbology', 'Astronomy', 'Ancient Runes', 'Defense Against the Dark Arts']
    target = 'Hogwarts House'
    X, y = LoadDataset(path, features, target)
    if len(X) != len(y):
        print("X and y size does not match")
        exit()
    X = np.array(X)
    #FillNan(X, np.nanmean(X, axis=0)) #FillNan(X, np.nanmedian(X, axis=0)) #FillNanKNN(X, X) #DropNan(X, y)
    FillNan(X, np.array([-1] * X.shape[0]))
    sample = np.array(X[0])
    X, mean, std = Normalize(X)
    coef = ((sample - mean) / std) / sample
    y = np.array(y)
    X = np.c_[[1] * X.shape[0], X]
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.1, random_state=21)
    if "-SGD" in sys.argv:
        logreg = LogisticRegression(optimizer='SGD',
                epoch=1400,
                lr=.115,
                random_state=42)
    elif "-BGD" in sys.argv:
        logreg = LogisticRegression(optimizer='BGD',
                epoch=11,
                lr=.11,
                random_state=5,
                batch_size=10)
    else:
        logreg = LogisticRegression(optimizer=None,
                epoch=70,
                lr=7)
    logreg.fit(X_train, y_train)
    #logreg.fit(X, y) # this yields less accuracy
    pred = logreg.predict(X_test)
    print("Test accuracy:", accuracy(pred, y_test))
    coef = np.insert(((sample - mean) / std) / sample, 0, 1)
    logreg.unnormalize_weights(coef)
    dump(logreg, 'logreg.model')


if __name__ == '__main__':
    main()
