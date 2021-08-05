import csv
import numpy as np

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


def train_test_split(X, y, test_size=0.2):
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
