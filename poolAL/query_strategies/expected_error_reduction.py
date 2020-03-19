#!/usr/bin/env python
# coding: utf-8

# In[1]:


## A query strategy that queries the sample that reduces the expected error the most

import numpy as np
from .core import QueryStrategy, Model
from .core.utils import zipit, entropy, sort_by_2nd
import copy
from pathos.multiprocessing import ProcessingPool as Pool
import threading


# In[ ]:


class ExpectedErrorReduction(QueryStrategy):
    '''
    EER algorithm.
    Idea: Chose the sample that when added to labeled data reduces the expected error the most.

    Parameters
    ----------
    dataset: {Dataset}

    model: {Model}

    depth: {int}
        We optimise the expected error after adding "depth" samples.
        Note: Computation time is of O(n_unlabeled**depth)!
        default = 1
    '''

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset)

        ## Setup parameters
        self.model = kwargs.pop('model', None)
        if not isinstance(self.model, Model):
            raise TypeError('model is not a Model object')

        self.depth = kwargs.pop('depth', 1)

        ## possible labels
        self.possible_labels = None

    def _eer(self, idx, dataset, depth):
        depth -= 1

        ## IMPORTANT copy wont work, must use deepcopy
        d = copy.deepcopy(dataset)

        ## Delete entry idx from the dataset
        ## Here again Deepcopy required!
        reduced_dataset = copy.deepcopy(d)
        reduced_dataset.delete_entry(idx)

        ## number of possible labels in the dataset
        nr_of_labels = len(self.possible_labels)

        ## initiate array
        uncertainty_after_adding = np.empty(nr_of_labels)

        for j in range(nr_of_labels):
            ## update dataset with every possible label
            d.update(idx, self.possible_labels[j])

            ## Train the model
            self.model.train(d)

            if depth >= 1:
                ## Calculate probabilities
                X_ids, X = reduced_dataset.get_unlabeled_entries()
                pred = self.model.predict_proba(X)
                l = len(X)
                uncertainty_after_adding_j = np.asarray(list(map(self._eer, X_ids, l*[reduced_dataset], l*[depth])))
                uncertainty_after_adding[j] = min(np.inner(pred, uncertainty_after_adding_j).diagonal())

            else:
                X_ids, X = reduced_dataset.get_unlabeled_entries()
                pred = self.model.predict_proba(X)
                u = 0
                for k in pred:
                    u += entropy(k)
                uncertainty_after_adding[j] = u
        return uncertainty_after_adding


    def make_query(self, size = 1):

        ## quit if nr_unlabeled_samples = 1
        if self.dataset.len_unlabeled() == 1:
            return self.dataset.get_unlabeled_entries()[0].astype(int)

        ## Set the possible labels
        self.possible_labels = list(set(self.dataset.get_labeled_entries()[1]))

        ## Train the model
        self.model.train(self.dataset)

        ## Get probabilities
        X_ids, X = self.dataset.get_unlabeled_entries()
        pred = self.model.predict_proba(X) # pred.shape = (n_unlabeled, nr_of_labels)

        ## Setup pool for cpu parallelisation
        p = Pool(threading.active_count(), maxtasksperchild = 1000)

        ## nr of unlabeled samples -> len(X)

        ## Get uncertainty after adding every sample with every label
        total = np.asarray(p.map(self._eer, X_ids, len(X)*[self.dataset], len(X)*[self.depth]))
        # total.shape = (n_unlabeled, nr_of_labels)

        ## Close the Pool again
        p.close()
        p.join()
        p.clear()

        ## Get the total uncertainty of one sample after adding a label weighted by the labels probability
        total = np.inner(pred, total,).diagonal() # total.shape = (n_unlabeled,)

        ## Zip it
        total = zipit(X_ids, total)

        ## Sort it
        results = sort_by_2nd(total, 'min')

        return results[:size,0].astype(int)

    def confidence(self):
        pass
