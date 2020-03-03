#!/usr/bin/env python
# coding: utf-8

# In[7]:


from abc import ABC, abstractmethod


# In[8]:


class QueryStrategy(ABC):
    '''
    Pool-based query strategy
    
    '''
    
    def __init__(self, dataset, **kwargs):
        self._dataset = dataset
        
        ## Sanity check for dataset
        #if not isinstance(self.dataset, Dataset):
         #   raise TypeError(
          #      'dataset is not a Dataset object'
           # )
        
    @property
    def dataset(self):
        return self._dataset
    
    def update(self, entry_id, label):
        '''
        Update the query strategy after newly labeled sample.
        
        Parameters
        ----------
        entry_id: int
            Index of newly labeled sample
        label: int
            Label of it
        '''
        self.dataset.update(entry_id, label)
        
        #Here refit
        
    @abstractmethod
    def confidence(self):
        '''
        Gives the confidence of the query in every entry in percentage.
        
        Returns
        -------
        np.array, shape = (n_entries)
        '''
        pass
        
    @abstractmethod
    def make_query(self):
        '''
        Return the index of the sample to be queried and labeled. 
        
        Returns
        -------
        entry_id: int
            The index of an unlabeled sample out of dataset
        '''
        pass


# In[ ]:




