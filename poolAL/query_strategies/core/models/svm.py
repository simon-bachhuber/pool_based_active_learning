#!/usr/bin/env python
# coding: utf-8

# In[14]:


from .model import Model
from sklearn.svm import SVC


# In[12]:


class SVM(Model):
    '''
    SVM with default probability = 1
    '''
    def __init__(self, *args, **kwargs):
        self.model = SVC(probability = kwargs.pop('probability', True), *args, **kwargs)

    def train(self, dataset):
        X, y = dataset.get_labeled_entries()
        self.model.fit(X, y)

    def predict(self, feature):
        return self.model.predict(feature)

    def predict_proba(self, feature):
        return self.model.predict_proba(feature)

    def score(self, dataset):
        X, y = dataset.get_labeled_entries()
        return self.model.score(X, y)


# In[ ]:
