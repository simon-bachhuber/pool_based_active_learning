#!/usr/bin/env python
# coding: utf-8

# In[1]:


from abc import ABC, abstractmethod


# In[3]:


class Model(ABC):
    '''
    Classification Model
    '''
    @abstractmethod
    def train(self, dataset, *args, **kwargs):
        '''
        Train a model given a dataset

        Parameters
        ----------
        dataset: Dataset object
        '''
        pass

    @abstractmethod
    def predict(self, feature, *args, **kwargs):
        '''
        Predict the class label of input

        Parameters
        ----------
        feature: np array, shape = (n_samples, n_features)

        Returns
        -------
        y_pred: np array, shape = (n_samples)
        '''
        pass

    @abstractmethod
    def score(self, testing_dataset, *args, **kwargs):
        """Return the mean accuracy on the test dataset
        Parameters
        ----------
        testing_dataset : Dataset object
            The testing dataset used to measure the perforance of the trained
            model.
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        pass

'''
    @abstractmethod
    def predict_proba(self, feature, *args, **kwargs):
        """Predict probability estimate for samples.
        Parameters
        ----------
        feature : array-like, shape (n_samples, n_features)
            The samples whose probability estimation are to be predicted.
        Returns
        -------
        X : array-like, shape (n_samples, n_classes)
            Each entry is the prabablity estimate for each class.
        """
        pass
'''

# In[ ]:
