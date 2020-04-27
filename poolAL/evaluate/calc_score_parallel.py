import copy
from ..query_strategies.core.utils import shuffle
from ..query_strategies.core import Dataset
import numpy as np
import time
from IPython.display import clear_output
from pathos.multiprocessing import ProcessPool as Pool
from pathos.multiprocessing import cpu_count

def CalcScoreParallel(_X, _y, qs, qs_kwargs, clf, n_labels_start, n_labels_end, n_runs, n_unique_labels_start, **kwargs):
    '''
    A cpu parallelized version of CalcScore. Runs on all available threads.
    Calculates the mean score of a list of query strategies (or a single qs) given a budget of labels
    Don't use it with warm started classifiers, use poolAL.evaluate.calc_score.CalcScore instead.


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
    random_state: {int}
        default = None


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

    # Label budget
    n_labels = n_labels_end - n_labels_start

    # Setup random_states to make it perfectly reproducable
    np.random.seed(kwargs.pop('random_state', None))
    random_state = np.array([np.random.randint(0, 2**32) for x in range(n_runs)])


    def run(_):

        nonlocal _X
        nonlocal _y


        result = np.zeros((N, n_labels, 2))

        # Shuffle
        X, y = shuffle(_X, _y, n_unique_labels_start, random_state = random_state[_])

        # Create Datasets
        data_train = Dataset(X[:n_labels_end], y[:n_labels_start]+(n_labels_end-n_labels_start)*[None])
        # Test pool is not larger than 300 samples
        data_test = Dataset(X[n_labels_end:n_labels_end+300], y[n_labels_end:n_labels_end+300])

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
                result[k,j,0] = j + n_labels_start

                # Make query and update
                idx = qs_inst[k].make_query()
                if not isinstance(idx, int):
                    idx = idx[0]
                data_train[k].update(idx, y[idx])

        return result

    # Averaging runs
    ## Setup pool for cpu parallelisation
    p = Pool(cpu_count(), maxtasksperchild = 1000)

    result = np.asarray(p.map(run, range(n_runs)))
    # take the Mean
    result = np.mean(result, axis = 0)

    ## Close the Pool again
    p.close()
    p.join()
    p.clear()

    return result
