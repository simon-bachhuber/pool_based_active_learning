import numpy as np
from .dataset import Dataset
from sklearn.utils import shuffle as sk_shuffle
import copy

def entropy(a):
    '''
    Calculates entropy of given discrete probability distribution.

    Parameters:
    ----------
    a: np.array, shape = (n_classes)

    Returns:
    float
    '''
    s = 0
    for i in a:
        if i not in [0]:
            s += -i*np.log(i)
    return s

def zipit(a,b):
    return np.array([[a[i], b[i]] for i in range(len(a))])

def unzipit(a):
    return np.array([x[0] for x in a]), np.array([x[1] for x in a])

def sort_by_2nd(a, max_min):
    '''
    Sorts an array of shape = (n_samples, 2) by second arguments.

    E.g.
    a = np.array([[1, 0.5],
                  [2, 0.1],
                   3, 0.8]])

    sory_by_2nd(a, 'max') ->
        np.array([[3, 0.8],
                  [1, 0.5],
                  [2, 0.1]])
    '''

    a = a.tolist()
    if max_min == 'max':
        reverse = True
    else:
        reverse = False
    a.sort(key=lambda x:x[1],reverse=reverse)
    return np.asarray(a)

def manhattan_metric(a,b):
    '''
    Your typical manhattan metric.

    Parameters
    ----------
    a, b: np.arrays of shape = (n_features,)


    Returns
    -------
    distance: float
    '''
    return np.sum(np.abs(a-b))

def euclidian_metric(a,b):
    '''
    Your typical euclidian metric.

    Parameters
    ----------
    a, b: np.arrays of shape = (n_features,)


    Returns
    -------
    distance: float
    '''
    return np.sqrt(np.sum((a-b)**2))

def shuffle(X, y, n_classes, random_state = None):
    '''
    Shuffles X, y in such a way that the first n_classes samples are all unique classes.

    X: {np.array}
    y: {list}

    '''
    d = Dataset(X, y)

    # The number of samples per class and unique labels
    _ = d.class_balance()
    labels, p = list(_.keys()), np.fromiter(_.values(), dtype = int)
    p = p/d.__len__()

    # Draw which classes are going to be initial classes
    np.random.seed(random_state)
    n_classes_labels = np.random.choice(labels, n_classes, replace = False, p=p)

    # Shuffle
    X, y = sk_shuffle(X, y, random_state = random_state)

    # Generate mask for permutation
    count = -1
    y = np.array(y)

    for l in n_classes_labels:

        # index of the drawn classes
        idx = y.tolist().index(l)

        permu = np.arange(d.__len__())
        count += 1

        permu[count] = idx
        permu[idx] = count

        # Do the permutation
        X = X[permu]
        y = y[permu]

    y = y.tolist()

    return X, y

def get_grid(X, n_grid):

    mi = np.min(X, axis = 0)
    ma = np.max(X, axis = 0)

    dim = X.shape[1]

    coor = []
    for i in range(dim):
        coor.append(np.linspace(mi[i], ma[i], n_grid))

    mesh = np.meshgrid(*coor)

    # Flatten the meshgrid
    for i in range(dim):
        mesh[i] = mesh[i].flatten()

    # Convert meshgrid into form (n_samples, n_features)
    a = copy.copy(mesh[0].reshape((mesh[0].shape+(1,))))
    a.fill('nan')

    for arr in mesh:
        a = np.concatenate((a, arr.reshape((arr.shape+(1,)))), axis = -1)

    return np.apply_along_axis(lambda a: a[1:], axis = -1, arr = a)
