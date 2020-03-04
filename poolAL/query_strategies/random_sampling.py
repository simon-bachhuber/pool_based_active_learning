#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from .core import QueryStrategy
import numpy as np


# In[ ]:


class RandomSampling(QueryStrategy):
    '''
    A random query strategy. 
    Makes query according to given random probability distribution.
    
    Parameter
    ----------
    dataset: Dataset object
    
    
    Methods
    -------
    .make_query(size = 1): Returns size number of entry ids to be queried.
    '''
    
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset)
            
    def make_query(self, size = 1):
        entries, _ = self.dataset.get_unlabeled_entries()
        np.random.shuffle(entries)
        return entries[:size]
    
    def confidence(self):
        pass


# In[ ]:




