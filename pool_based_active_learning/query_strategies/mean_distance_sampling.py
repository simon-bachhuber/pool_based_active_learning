#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from .core import QueryStrategy
import numpy as np
from .core.utils import sort_by_2nd, manhattan_metric


# In[ ]:


## A simple query strategy that either favors points with a low or high mean distance to (all) other sample points or just a subspace.
## The idea is that in a dense region a sample has a lower mean distance to other points than if it was in a sparse region.

class MeanDistanceSampling(QueryStrategy):
    '''
    Query strategy that querys according to samples mean distance to other points in data space.
    Two general approaches:
        - Low distance w.r.t. all samples or all unlabeled samples -> Choose most representative ones
        - High distance w.r.t. all labeled samples -> Choose samples in unexplored region

    Parameters
    ----------

    goal: {string}, either 'high' or 'low'
        Determines wether to query for samples with high or low mean distance
        default = 'low'

    space: {string}, either 'unlabeled', 'labeled' or 'all'
        Mean distance w.r.t. to what samples
        default = 'unlabeled'

    epsilon: {float}, (0,1]
        Greediness parameter. Amount in % of samples out of space to use for calculating mean distances. 
        default = 1
        
    metric: {fkt}
        Function that returns the distance between two sample points.

    Methods
    -------

    .make_query(size = 1): Returns size number of entry ids to query. 
        default = 1 
    '''

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset)

        # Set parameters
        self.goal = kwargs.pop('goal', 'low')
        self.space = kwargs.pop('space', 'unlabeled')
        self.epsilon = kwargs.pop('epsilon', 1)
        self.metric = kwargs.pop('metric', manhattan_metric)

        # Sanity checks
        if self.goal not in ['low', 'high']:
            raise ValueError('goal parameter is no valid option')

        if self.space not in ['labeled', 'unlabeled', 'all']:
            raise ValueError('space parameter is no valid option')

    def confidence(self):
        pass

    def make_query(self, size = 1):
        
        ## Stop when no more unlabeled samples
        if self.dataset.len_unlabeled() == 0:
            raise IterationError('Every sample in dataset is labeled')
        
        # Obtain the correct space
        # space is going to be the ids of all unlabeld, labeled or just all samples
        if self.space == 'labeled':
            space = self.dataset.get_labeled_entries_ids()
        elif self.space == 'unlabeled':
            space, _ = self.dataset.get_unlabeled_entries()
        elif self.space == 'all':
            space = np.array([x for x in range(self.dataset.__len__())])
            
        if self.epsilon is not 1:
            # Shrink the respective space in greedy strategy
            greedy_nr = round(self.epsilon*len(space))
            space = np.random.choice(space, size = greedy_nr, replace = False)
        
        ## Get all unlabeled ids
        unlabeled_ids, _ = self.dataset.get_unlabeled_entries()
        
        ## Calculate the mean distance
        dist_matrix = np.zeros((len(unlabeled_ids), 2))
        
        ## Iteration 
        nr = 0
        for idx in unlabeled_ids:
            ## Save the id in the dist_matrix
            dist_matrix[nr, 0] = idx
            for space_idx in space:
                dist = self.metric(self.dataset._X[idx], self.dataset._X[space_idx])
                dist_matrix[nr, 1] += dist
            nr += 1
            
        ## Take the mean
        dist_matrix[:, 1] = dist_matrix[:, 1]/len(space)
        
        ## Rank/Sort
        if self.goal == 'high':
            return sort_by_2nd(dist_matrix, 'max')[:size, 0].astype(int)
        elif self.goal == 'low':
            return sort_by_2nd(dist_matrix, 'min')[:size, 0].astype(int)

