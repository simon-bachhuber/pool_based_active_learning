import numpy as np
import matplotlib.pyplot as plt


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
            y.append(0)

    return X, y

def LinearDecisionBoundary(w = np.array([1, -1])):

    N = 400

    np.random.seed(1)
    X = np.random.rand(N, 2)

    y = []
    for sample in X:
        y.append(np.sign(np.einsum('i,i', sample, w)))

    return X, y 
