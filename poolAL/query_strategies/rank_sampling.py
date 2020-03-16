#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from .core import QueryStrategy
from .core.utils import zipit, sort_by_2nd

## Combines several query strategies together by summing the rank of every unlabeled sample in all the query strategies. The lowest overall rank is queried next.


# In[ ]:


class RankSampling(QueryStrategy):
    '''
    Combines several QueryStrategies by summing up the ranking.
    Note: All QueryStrategies share the same Dataset!

    Parameters
    ----------

    query_strategy: {list}
        The list of QueryStrategy objects that contributes in the ranking

    ranking: {string}, either 'linear' or 'exponential'
        The scaling of the ranking before it is summed up.
        E.g. Given two QS's with 'linear' the combination rank 1 and rank 6 is just as good as rank 2 and rank 5.
        With 'exponential' rank 1 and rank 6 will win usually. (depending on alpha)
        default = 'linear'

    alpha: {float}, > 0
        Scaling parameter of 'exponential' ranking
        default = 0.1


    Methods
    -------

    .make_query(size = 1)

    '''

    def __init__(self, _ = None, **kwargs):

        # Setup the parameters
        self.qs = kwargs.pop('query_strategy', None)

        self.ranking = kwargs.pop('ranking', 'linear')

        self.alpha = kwargs.pop('alpha', 0.1)

        # Number of QS objects
        self.N = len(self.qs)

        # Number of samples
        self.n = self.qs[0].dataset.__len__()


    def make_query(self, size = 1, _exact_results = 0):
        # Number of unlabeled entries in dataset
        l = self.qs[0].dataset.len_unlabeled()

        # Get all query orders
        query_order = np.zeros(l)
        for  qs in self.qs:
            query_order = np.vstack((query_order, qs.make_query(size = l)))
        query_order = query_order[1:].astype(int)

        # Sum them up
        ranking = np.zeros(self.n)
        for i in range(l):
            tmp = query_order[:, i]
            for j in tmp:
                if self.ranking == 'linear':
                    ranking[j] += (i+1) # i+1 is the rank.. Entry 0 is rank 1
                else:
                    ranking[j] += -np.exp(-self.alpha * (i+1))

        # Write them in format [entry_id, sum of ranks]
        a = np.zeros(2)
        for i in range(self.n):
            if ranking[i] not in [0]:
                a = np.vstack((a, np.array([i, ranking[i]])))
        ids_ranks = a[1:]

        # Sort them, the lower total rank the better
        result = sort_by_2nd(ids_ranks, 'min')

        if _exact_results == 1:
            return result
        else:
            return result[:size, 0].astype(int)

    def confidence(self):
        '''
        Returns
        -------

        {np.array} of shape = (n_unlabeled, 2)
            [[entry_id, confidence in percentage],[...]]
        '''
        result = self.make_query(_exact_results = 1)

        ## Exponential ranking, Gibbs measure
        if result[0,1] < 0:
            result[:, 1] = np.exp(-result[:,1]/self.alpha)
            result[:, 1] = result[:,1] / np.sum(result[:,1])

        ## Linear ranking
        else:
            result[:, 1] = 1/result[:,1]
            result[:, 1] = result[:,1] / np.sum(result[:,1])

        return result
