#!/usr/bin/env python3

from logistic_regression import LogisticRegression
from utils import load
from utils import LoadDataset
from utils import FillNan
from utils import save_predictions
import numpy as np


def main():
    logreg = load('logreg.model')
    path = './data/dataset_test.csv'
    output_path = './data/houses.csv'
    features = ['Herbology', 'Astronomy', 'Ancient Runes']
    X, _ = LoadDataset(path, features)
    X = np.array(X)
    X = np.c_[[1] * X.shape[0], X]
    #FillNan(X, np.nanmean(X, axis=0))
    FillNan(X, np.array([0] * X.shape[0]))
    pred = logreg.predict(X)
    save_predictions(output_path, pred)


if __name__ == '__main__':
    main()
