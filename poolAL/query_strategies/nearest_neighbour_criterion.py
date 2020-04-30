from .core.utils import zipit, sort_by_2nd, euclidian_metric
from .core import QueryStrategy
import numpy as np

class NearestNeighbourCriterion(QueryStrategy):
    '''
    Representative query strategy. Its goal is to maximize the ability of the queried data
    to represent the unlabeled data. The scheme is the following:

        - For every unlabeled sample x, calculate the smallest distance to a labeled sample x',
          this is the Nearest Neighbour distance NND
          NND(x, L) = min_{x' from L} |x-x'|

        - Define NNC as the sum over NND for every x from U
          NNC(U, L) = Sum_{x from U} NND(x, L)

        - Chose the unlabeled sample that minimizes NNC after it is added to the labeled pool
          x_queried = argmin_{x from U} NNC(U \ x, L + x)

    NNC is a measure how well the labeled data represents the unlabeled data, the lower the better.

    Parameters
    ----------
    dataset: {poolAL.query_strategies.core.Dataset}

    metric: {fkt}
        function that assigns two points in data space a scalar value.
        default = euclidian


    Methods
    -------
    .make_query(size = 1)


    References
    ----------
        [1] Towards practical active learning for classification. Yazhou Yang. 2018.
            Page 93-94.
    '''
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset)

        # Get a metric
        self.metric = kwargs.pop('metric', euclidian_metric)

        # Calculate two point distance matrix
        self.distance = self._get_distance()

        self.grid_distance = None

    def _get_distance(self):
        X = self.dataset._X

        # Number of samples
        n = len(X)

        d = np.zeros((n,n))
        for i in range(n):
            for j in range(i):
                temp = self.metric(X[i], X[j])
                d[i,j] = temp
                d[j,i] = temp

        return d

    def _get_scores(self):
        # Update two point matrix if samples were appended
        if self.dataset.modified_X:
            self.distance = self._get_distance()
            self.dataset.modified_X = False

        labeled_ids = self.dataset.get_labeled_entries_ids()
        unlabeled_ids, _ = self.dataset.get_unlabeled_entries()

        # Number of unlabeled samples
        n_unlabeled = len(unlabeled_ids)

        # Instantiate vector for scores
        score = np.zeros(n_unlabeled)

        for i in range(n_unlabeled):

            # Move one unlabeled to labeled
            temp_u = np.delete(unlabeled_ids, i)
            temp_l = np.append(labeled_ids, unlabeled_ids[i])

            for idu in temp_u:
                nnd = []
                for idl in temp_l:
                    nnd.append(self.distance[idu, idl])

                # Keep the smallest nnd
                score[i] += min(nnd)

        # Zipit
        score = zipit(unlabeled_ids, score)

        return score

    def make_query(self, size = 1):
        score = self._get_scores()

        # Sort
        results = sort_by_2nd(score, 'min')

        return results[:size, 0].astype(int)


    def confidence(self):
        '''
        The lower the better.
        '''
        return self._get_scores()[:,1]

    def confidence_grid(self, grid_X):
        '''
        Parameters
        ----------

        grid_X: {np.array} of form (n_grid_points, n_features)
        '''
        if self.grid_distance is None:
            # Calculate distance matrix
            X = np.vstack((self.dataset._X, grid_X))

            # Number of samples
            n = len(X)

            d = np.zeros((n,n))
            for i in range(n):
                for j in range(i):
                    temp = self.metric(X[i], X[j])
                    d[i,j] = temp
                    d[j,i] = temp

            self.grid_distance = d


        labeled_ids = self.dataset.get_labeled_entries_ids()
        unlabeled_ids, _ = self.dataset.get_unlabeled_entries()
        grid_ids = np.arange(len(self.dataset._X), self.grid_distance.shape[0])

        # Number of grid points
        n = len(grid_X)

        # Instantiate vector for scores
        score = np.zeros(n)

        for i in range(n):

            # Move one grid point to labeled
            temp_l = np.append(labeled_ids, grid_ids[i])

            for idu in unlabeled_ids:
                nnd = []
                for idl in temp_l:
                    nnd.append(self.grid_distance[idu, idl])

                # Keep the smallest nnd
                score[i] += min(nnd)

        return score
