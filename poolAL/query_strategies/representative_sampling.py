#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Query strategy that gives you samples from a high or low density region w.r.t. labeled, unlabeled or full sample space
## Density is calculated either with a constant sphere in data space or k means clustering
## Similar to MeanDistanceSampling

from .core import QueryStrategy
from sklearn.cluster import KMeans
from .core.utils import manhattan_metric, zipit, sort_by_2nd
import numpy as np


# In[ ]:


class RepresentativeSampling(QueryStrategy):
    '''
    QueryStrategy designed to find samples most representative of other samples - meaning samples in a high density region
    Density is estimated using either a sphere with constant radius or kMeans clustering algorihm.

    Possible Strategies:
        - high w.r.t. unlabeled or full space
        - low w.r.t. labeled space

    Parameters
    ----------

    goal: {string}, either 'high' or 'low'
        Wether to return samples in a high or low density region
        default = 'high'

    space: {string}, out of ['labeled', 'unlabeled', 'all']
        The density of the samples in w.r.t. to a data space containing labeled, unlabeled or all samples
        default = 'unlabeled'

    method: {string}, either 'sphere' or 'cluster'
        default = 'sphere'

    metric: {fkt}
        A metric function.
        default = Manhattan metric


    Methods
    -------

    .make_query(size = 1): {np.array}, dtype = int, shape = (size)
        Returns the entry id's to query next


    '''

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset)

        ## Setting up parameters
        self.max_min = kwargs.pop('goal', 'high')
        if self.max_min not in ['high', 'low']:
            raise ValueError('max_min parameter is not a valid option')

        self.space = kwargs.pop('space', 'unlabeled')
        if self.space not in ['unlabeled', 'labeled', 'all']:
            raise ValueError('space parameter is not a valid option')

        self.method = kwargs.pop('method', 'sphere')
        if self.method not in ['cluster', 'sphere']:
            raise ValueError('method parameter is not a valid option')

        self.metric = kwargs.pop('metric', manhattan_metric)

        ## Initiate kMeans
        self.nr_of_clusters = self.dataset.get_num_of_labels()

        if self.method == 'cluster':
            self.kMeans = KMeans(n_clusters = self.nr_of_clusters, random_state= 2)
            self.kMeans.fit(self.dataset._X)

        ## Initiate Sphere method
        if self.method == 'sphere':
            ## Calc distance matrix
            self.dist_matrix = self._calc_dist_matrix()

            ## Calc mean distance
            self.mean_dist = self._calc_mean_dist()

    def make_query(self, size =1):

        # Sphere density
        if self.method == 'sphere':
            results = self._score_sphere()

            # Sort them accordingly
            if self.max_min == 'high':
                results = sort_by_2nd(results, 'max')
            else:
                results = sort_by_2nd(results, 'min')

            return results[:size, 0].astype(int)

        # kMeans clustering
        if self.method == 'cluster':
            results = self._score_cluster()

            # Only keep main cluster, so smallest distance to any cluster center
            results[:,1] = np.array([np.min(e) for e in results[:,1]])

            # Sort them accordingly
            # if high density then low distance to cluster
            if self.max_min == 'high':
                results = sort_by_2nd(results, 'min')
            else:
                results = sort_by_2nd(results, 'max')

            return results[:size, 0].astype(int)



    def _score_cluster(self):

        ## Adjust n_components if required
        nr = self.dataset.get_num_of_labels()
        nr_of_clusters_changed = False

        if nr is not self.nr_of_clusters:
            self.nr_of_clusters = nr
            self.kMeans = KMeans(n_clusters= nr)
            nr_of_clusters_changed = True

        ## Check that n_unlabeled_samples >= n_clusters
        if self.space == 'unlabeled' and self.nr_of_clusters > self.dataset.len_unlabeled():
            raise ValueError('There are more clusters than unlabeled samples.' +
                             'Chose different space parameter.'
            )

        ## Fit the kMeans to the space if required
        if self.space == 'all':
            X, _ = self.dataset.get_entries()
        elif self.space == 'unlabeled':
            _, X = self.dataset.get_unlabeled_entries()
        else:
            X, _ = self.dataset.get_labeled_entries()

        if self.space == 'all' and nr_of_clusters_changed:
            self.kMeans.fit(X)
        else:
            self.kMeans.fit(X)


        ## Transform to cluster space
        X_ids, X = self.dataset.get_unlabeled_entries()
        X = self.kMeans.transform(X)

        ## Zipit
        return zipit(X_ids, X)

    def _score_sphere(self):
        ## Grab the unlabeled sample ids to calc their density in space
        X_ids = self.dataset.get_unlabeled_entries()[0]

        ## Determine the space
        if self.space =='all':
            space_ids = np.arange(self.dataset.__len__())
            density = np.zeros(len(X_ids))
        elif self.space == 'unlabeled':
            space_ids = self.dataset.get_unlabeled_entries()[0]
            density = np.zeros(len(X_ids))
        else:
            space_ids = self.dataset.get_labeled_entries_ids()
            density = np.ones(len(X_ids))

        ## Go through them and check if sample in space are within sqhere with radius mean distance
        count = 0

        for id1 in X_ids:
            for id2 in space_ids:
                if self.dist_matrix[id1, id2] < self.mean_dist:
                    density[count] += 1
            count += 1

        ## Zipit
        return zipit(X_ids, density)


    def _calc_dist_matrix(self):
        X = self.dataset.get_entries()[0]

        ## number of total samples
        l = len(X)

        ## Calc the distance between all pairs possible. Note: two point distance matrix is SYMMETRIC
        m = np.zeros((l, l))
        for i in range(l):
            for j in range(i):
                d = self.metric(X[i], X[j])
                m[i,j] = d
                m[j,i] = d

        return m

    def _calc_mean_dist(self):
        l = self.dist_matrix.shape[0]
        d = 0

        for i in range(l):
            for j in range(i+1,l):
                d += self.dist_matrix[i,j]

        return d/ ((l**2 - l)/2) # (l**2-l)/2 is number of offdiagonal elements in a matrix of shape (l,l)

    def confidence(self):
        pass


# In[ ]:
