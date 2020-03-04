import copy
from sklearn.utils import shuffle
from .dataset import Dataset
import numpy as np
from IPython.display import clear_output

def Scorer(X, y, qs, qs_kwargs, clf, n_labels_start, n_labels_end, n_runs, n_unique_labels_start):
    '''
    Calculates the mean score of a list of query strategies (or a single qs) given a budget of labels
    
    
    Parameters
    ----------
    
    X: {np.array}
    y: {list}
    qs: {list} of QueryStrategy objects
    qs_kwargs: {list} of Dictionaries
    clf: {Model}
    n_labels_start: {int}
    n_labels_end: {int}
    n_runs: {int}
    n_unique_labels_start: {int}
    
    
    Returns
    -------
    np.array of shape (n_query_strategies, n_labels, 2)
    
    '''
    # Number of query strategies
    N = len(qs)
    
    # Result array
    result = np.zeros((N, n_labels_end - n_labels_start, 2))
    
    n_labels = result.shape[1]
    
    for i in range(n_runs):
        clear_output(wait = True)
        print('Progress:',round((i+1)/n_runs*100,3),'%')
        # Shuffle until every class is present in initial data
        escape = 0
        while True:
            escape += 1
            X, y = shuffle(X, y)
            if len(set(y[:n_labels_start])) >= n_unique_labels_start:
                # Found a permutation
                break
            # Raise Error if its too hard
            if escape > 100:
                raise Exception('Ensuring that enough classes are present initially is too hard')
        
        # Create Datasets
        data_train = Dataset(X[:n_labels_end], y[:n_labels_start]+(n_labels_end-n_labels_start)*[None])
        data_test = Dataset(X[n_labels_end:], y[n_labels_end:])
        
        # Create copies of trainings data
        data_train = [copy.deepcopy(data_train) for i in range(N)]
        
        # Instantiate query strategies
        qs_inst = []
        for p in range(N):
            qs_inst.append(qs[p](data_train[p], **qs_kwargs[p]))
                           
        
        for j in range(n_labels):
            for k in range(N):
                # Score 
                clf.train(data_train[k])
                result[k,j,1] += clf.score(data_test)
                if i == 0:
                    result[k,j,0] = j + n_labels_start
                
                # Make query and update
                idx = qs_inst[k].make_query()
                if not isinstance(idx, int):
                    idx = idx[0]
                data_train[k].update(idx, y[idx])
                
    # Take the mean
    result[:,:,1] = result[:,:,1]/n_runs
    return result
        
    