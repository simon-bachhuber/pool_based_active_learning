#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# A query strategy aimed to find uncertain samples. Idea: Query samples that are in between two cluster centers, i.e., samples where kMeans clustering is most unsure of the samples cluster affinity.

from sklearn.cluster import KMeans
from .core import QueryStrategy
import numpy as np
from .core.utils import zipit, sort_by_2nd


# In[ ]:


class ClusterMarginSampling(QueryStrategy):
    '''
    *doc*

    Parameters
    ----------

    space: {string}, either 'full' or 'labeled'
        kMeans algorithm calculates his n_classes cluster centers based on either all samples or all labeled samples
    '''

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset)

        self.space = kwargs.pop('space', 'full')

        # Nr of cluster centers
        self.nr_of_clusters = self.dataset.get_num_of_labels()

        # kMeans algorithm
        self.kMeans = KMeans(n_clusters = self.nr_of_clusters, random_state= 2)
        self.kMeans.fit(self.dataset._X)

    def _get_scores(self):
        if self.space == 'labeled':
            X = self.dataset.get_labeled_entries()[0]
        elif self.space == 'full':
            X = self.dataset.get_entries()[0]

        # Check if number of clusters has changed and fit if yes
        if self.dataset.get_num_of_labels() != self.nr_of_clusters or self.dataset.modified_X:
            self.dataset.modified_X = False
            self.nr_of_clusters = self.dataset.get_num_of_labels()
            self.kMeans = KMeans(n_clusters = self.nr_of_clusters, random_state= 2)
            self.kMeans.fit(X)

        # Otherwise only fit if space is 'labeled'
        elif self.space == 'labeled':
            self.kMeans.fit(X)

        # Get unlabeled samples and ids
        X_ids, X = self.dataset.get_unlabeled_entries()

        # Transform to cluster space
        X = self.kMeans.transform(X)

        # Calc the margin
        X.sort()
        X_sorted = np.array([e[1]-e[0] for e in X])
        result = zipit(X_ids, X_sorted)

        return result


    def make_query(self, size = 1):
        result = self._get_scores()
        # Sort by 2nd, the lower the better
        result = sort_by_2nd(result, 'min')

        return result[:size, 0].astype(int)

    def confidence(self):
        '''
        The lower the better
        '''

        result = self._get_scores()

        return result[:,1]

    def confidence_grid(self, grid_X):
        '''
        The lower the better
        '''

        # This just fits the kMean alg
        _ = self._get_scores()

        X = self.kMeans.transform(grid_X)

        X.sort()
        return np.array([e[1]-e[0] for e in X])
