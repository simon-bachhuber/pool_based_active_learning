#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
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
    d = 0.0
    a, b = a.astype(float), b.astype(float)

    for i in range(len(a)):
        d += abs(a[i]-b[i])
    return d


# In[ ]:
