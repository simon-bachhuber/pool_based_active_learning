import copy
from sklearn.utils import shuffle
from .dataset import Dataset
import numpy as np
import time
from IPython.display import clear_output

def CalcScore(X, y, qs, qs_kwargs, clf, n_labels_start, n_labels_end, n_runs, n_unique_labels_start):
    '''
    Calculates the mean score of a list of query strategies (or a single qs) given a budget of labels


    Parameters
    ----------

    X: {np.array} of shape (n_samples, n_features)
        Input data
    y: {list}
        labels
    qs: {list} of poolAL.query_strategies.core.query_strategy.QueryStrategy objects
        List of query strategies that should be used
        NOTE: If a committee QueryStrategy is included in qs, then the repspective kwargs dictionary in
        qs_kwargs must include key 'query_strategy'. Its value must be a tuple of the form:
        ({list} of QueryStrategy objects, {list} of kwargs-Dictionaries)
        For example see bottom.
    qs_kwargs: {list} of Dictionaries
        Each query strategies repspective kwargs
    clf: {poolAL.query_strategies.core.models.model.Model}
        The classifier used to calculate the test scores
    n_labels_start: {int}
        Number of initially available labels
    n_labels_end: {int}
        Number of labels when finished
    n_runs: {int}
        Number of runs for averaging
    n_unique_labels_start: {int}
        Number of unique labels in initial labels


    Returns
    -------
    {np.array} of shape (len(qs), n_labels_end-n_labels_start, 2)
        Mean test score calculated on the remaining samples. So n_test_samples = len(X)-n_labels_end.
        Axis 0 is the different query strategies
        Axis 1 is the different number of labels
        Axis 2 is of the form (current_n_labels, current_test_score)


    Example
    -------

    X, y = load_iris(return_X_y=True)
    y = y.tolist()
    clf = SVM(kernel = 'linear', random_state = 1)

    deal_qs = [ClusterMarginSampling, UncertaintySampling]
    deal_qs_kwargs = [{'space':'full'}, {'model': clf}]

    qs = [RandomSampling, ClusterMarginSampling, UncertaintySampling, DynamicEnsembleActiveLearning]
    qs_kwargs = [{}, {'space':'full'}, {'model': clf}, {'query_strategy': (deal_qs, deal_qs_kwargs), 'model': clf, 'T':97}]

    CalcScore(X, y, qs, qs_kwargs, clf, 3, 100, 200, 3)

    '''
    # Number of query strategies
    N = len(qs)

    # Result array
    result = np.zeros((N, n_labels_end - n_labels_start, 2))

    n_labels = result.shape[1]

    t1 = time.time()
    t2 = 0

    for i in range(n_runs):

        # End timer
        t2 += time.time()-t1
        # Start timer
        t1 = time.time()

        if i != 0:
            clear_output(wait = True)
            print('Progress:',round((i+1)/n_runs*100,3),'%')
            print('Finished in approximately',(t2/60/i)*(n_runs-i),'minutes.')

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
        for j in range(N):

            # Instantiate the query_strategy list for Committee's
            if 'query_strategy' in qs_kwargs[j]:
                temp_kwargs = copy.deepcopy(qs_kwargs[j])
                tup = temp_kwargs.pop('query_strategy')
                committee_qs_inst = [tup[0][x](data_train[j], **tup[1][x]) for x in range(len(tup[0]))]
                temp_kwargs.update({'query_strategy': committee_qs_inst})
                qs_inst.append(qs[j](data_train[j], **temp_kwargs))

            else:
                qs_inst.append(qs[j](data_train[j], **qs_kwargs[j]))


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
