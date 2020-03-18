#!/usr/bin/env python
# coding: utf-8

# In[49]:


from .core import QueryStrategy, Model, Dataset
#import import_ipynb
from .core.utils import entropy, zipit, sort_by_2nd
import numpy as np


# In[24]:


class UncertaintySampling(QueryStrategy):
    '''
    A simple heterogeneity query.
    Query the sample with least confidence.
    
    Parameters
    ----------
    dataset: Dataset object
    
    model: Model object
        The Model to base your confidence on
        
    method: {string}, either 'lc', 'entropy' or 'margin'
        default = 'lc'
        
        Say p is class probability distro of one sample, then 
        
        -lc -> 1-max(p) (query the highest)
        -entropy -> entropy(p) (query the highest)
        -margin -> max_1(p)-max_2(p) (query the lowest)
            max_2 means second highest.
            
    record_scores: {string}, either True or False
        Wether to record scores during querying.
        default = False
        
    test_dataset: Dataset object, required if record_scores is True
    
    Methods
    -------
    .make_query(size = 1):
        Returns: np.array of shape = (size, dtype =int) 
            size number of entry ids to be queried.
        
        
    '''
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset)
        
        self.model = kwargs.pop('model', None)
        
        ## Sanity checks of model
        if self.model is None:
            raise TypeError(
                'Missing require keyword-only argument: model'
            )
        if not isinstance(self.model, Model):
            raise TypeError(
                'model keyword must be a Model object'
            )
        self.model.train(self.dataset)
        
        self.method = kwargs.pop('method', 'lc')
        
        ## Sanity checks of method
        if self.method not in ['lc', 'entropy', 'margin']:
            raise TypeError(
                'Supported methods are [lc, entropy, margin]'
                           )
            
        ## Set verbose mode
        self.record_scores = kwargs.pop('record_scores', False)
        if self.record_scores not in [True, False]:
            raise ValueError('Record_scores must be either True or False')
        
        if self.record_scores == True:
            self.test_dataset = kwargs.pop('test_dataset', None)
            if self.test_dataset is None:
                raise TypeError('If record_scores is True, then test_dataset must be given a Dataset object')
        self.current_score = None
        
    def _get_scores(self):
        d = self.dataset
        self.model.train(d)
        unlabeled_ids, unlabeled_samples = d.get_unlabeled_entries()
        
        pred = self.model.predict_proba(unlabeled_samples)
        
        if self.method == 'lc':
            return zipit(unlabeled_ids, 1-np.array([max(e) for e in pred]))
        
        if self.method == 'entropy':
            return zipit(unlabeled_ids, np.array([entropy(e) for e in pred]))
        
        if self.method =='margin':
            pred = np.sort(pred)
            return zipit(unlabeled_ids, np.array([e[-1]-e[-2] for e in pred]))
        
    def confidence(self):
        pass
        
    def make_query(self, size = 1):
        scores = self._get_scores()
        
        ## Record scores if demanded
        if self.record_scores == True:
            self.current_score = np.array([self.dataset.len_labeled(), self.model.score(self.test_dataset)])
        
        ## Decide wether to sort by highest or lowest
        if self.method == 'margin':
            scores = sort_by_2nd(scores, 'min')
        else:
            scores = sort_by_2nd(scores, 'max')
        
        return scores[:size, 0].astype(int)
            

