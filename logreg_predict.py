#!/usr/bin/env python3

from logistic_regression import LogisticRegression
from utils import load


def main():
    logreg = load('logreg.model')
    print(logreg.weights)


if __name__ == '__main__':
    main()
