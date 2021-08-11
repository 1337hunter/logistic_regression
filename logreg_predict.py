#!/usr/bin/env python3

from logistic_regression import LogisticRegression
from utils import load
from utils import LoadDataset
from utils import FillNan
from utils import FillNanKNN
from utils import save_predictions
import numpy as np
import math

def main():
    logreg = load('logreg.model')
    path = './data/dataset_test.csv'
    output_path = './data/houses.csv'
    features = ['Herbology', 'Astronomy', 'Ancient Runes', 'Defense Against the Dark Arts']
    X, _ = LoadDataset(path, features)
    X_train, _ = LoadDataset("./data/dataset_train.csv", features)
    X = np.array(X)
    X_train = np.array(X_train)
    X = np.c_[[1] * X.shape[0], X]
    X_train = np.c_[[1] * X_train.shape[0], X_train]
    #FillNan(X, np.nanmean(X_train, axis=0))
    FillNan(X, np.array([-1] * X.shape[0]))
    #FillNan(X, np.nanmedian(X_train, axis=0))
    #FillNanKNN(X, X_train)
    pred = logreg.predict(X)
    save_predictions(output_path, pred)


if __name__ == '__main__':
    main()
