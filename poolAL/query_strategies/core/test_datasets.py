import numpy as np
import matplotlib.pyplot as plt
from .utils import zipit, shuffle
from .dataset import Dataset

def BinaryCircle():
    ## Generate a test dataset, binary classification with circular desicion boundary. Uncertainty Sampling excels here.

    # Number of samples
    N = 400
    # Radius of circle
    r = 0.33

    X = np.random.rand(N, 2)
    y = []

    # Define a circle function
    circle = lambda x1, x2: (x1-0.5)**2 + (x2-0.5)**2

    # Calculate labels
    for sample in X:
        if circle(sample[0], sample[1]) <= r**2:
            y.append(1)
        else:
            y.append(-1)

    return X, y

def data_aux_scoure(n_primary = 20, n_aux = 280, n_test = 300, acc_aux = 0.7, overlap = 0.1):

    N = int((n_primary + n_aux + n_test)/2)

    mean = [0, 1]
    cov = [[overlap, 0],[0, overlap]]
    x1, x2 = np.random.multivariate_normal(mean, cov, N).T
    mean = [1, 0]
    x11, x21 = np.random.multivariate_normal(mean, cov, N).T

    x1 = np.hstack((x1, x11))
    x2 = np.hstack((x2, x21))

    data = zipit(x1, x2)

    y = N*[1]+N*[-1]

    data, y = shuffle(data, y, 2)

    # Falsify data
    mask = np.random.choice(np.arange(n_primary, n_aux+n_primary), replace = False, size = int((1-acc_aux)*n_aux))
    y = np.array(y)
    y[mask] = y[mask]*-1

    train = Dataset(data[:(n_primary+n_aux)], y[:(n_primary+n_aux)])
    test = Dataset(data[-n_test:], y[-n_test:])

    return train, test

def BinaryCircleMargin():
    ## Generate a test dataset, binary classification with circular desicion boundary. Uncertainty Sampling excels here.

    # Number of samples
    N = 400
    # Radius of circle
    r = 0.33

    X = np.random.rand(N, 2)
    y = []

    # Define a circle function
    circle = lambda x1, x2: (x1-0.5)**2 + (x2-0.5)**2

    d = []
    for i in range(len(X)):
        if circle(X[i,0], X[i,1]) >= r**2 and circle(X[i,0], X[i,1]) <(r+0.1)**2:
            d.append(i)
    d = np.array(d)
    X = np.delete(X, d, axis = 0)

    # Calculate labels
    for sample in X:
        if circle(sample[0], sample[1]) <= r**2:
            y.append(1)
        else:
            y.append(-1)

    return X, y

def LinearDecisionBoundary(w = np.array([1, -1])):

    N = 400

    np.random.seed(1)
    X = np.random.rand(N, 2)

    y = []
    for sample in X:
        y.append(np.sign(np.einsum('i,i', sample, w)))

    return X, y
