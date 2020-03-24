#!/usr/bin/env python
# coding: utf-8

# In[2]:


## A simple query by committee strategy, the sample gets queried
## where a collection of classifiers disagrees the most on
from .core import QueryStrategy, Model
import numpy as np
from .core.utils import entropy, sort_by_2nd, zipit


# In[ ]:

def kl_div(a, b):
    shap = a.shape
    out = np.empty(shap)
    for i in range(shap[0]):
        for j in range(shap[1]):
            for k in range(shap[2]):
                x, y = a[i,j,k], b[i,j,k]
                if x >0 and y >0:
                    out[i,j,k] = x*np.log(x/y)
                elif x==0 and y >=0:
                    out[i,j,k] = 0
                else:
                    out[i,j,k] = NaN
    return out


class QueryByCommittee(QueryStrategy):
    '''
    Query by committee strategy. A bundle of classifiers vote on which sample they disagree the most.

    Parameters
    ----------

    dataset: {Dataset}

    committee: {list}
        list of Model objects. The classifier committee

    method: {string}
        Either vote for vote entropy or kl for kullback-leibner divergenz
        default = 'vote'

    bagging: {bool}
        Wether to use bootstrapping with replacement to create 'unique' datasets for every Model object.
        default = True


    Methods
    -------

    .make_query(size = 1)
        Returns {np.array} of shape =(size). The entry id's
    '''
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset)

        ## Setup our parameters
        self.committee = kwargs.pop('committee', None)
        for clf in self.committee:
            if not isinstance(clf, Model):
                raise TypeError('Every model in the committee must be a Model object')

        self.method = kwargs.pop('method', 'vote')

        ## Number of classifiers
        self.n = len(self.committee)

        ## Bagging
        self.bagging = kwargs.pop('bagging', True)

    def teach_committee(self):

        ## Wether to train models on bootstrapped data ...
        if self.bagging == True:

            ## nr of labels in dataset
            nr = self.dataset.len_labeled()
            for student in self.committee:
                bag = self.dataset.labeled_uniform_sample(nr)
                student.train(bag)

        ## or all on same dataset
        if self.bagging == False:
            for student in self.committee:
                student.train(self.dataset)

    def make_query(self, size =1):
        ## Vote entropy
        if self.method == 'vote':
            return self._vote_entropy(size)

        ## Kullback-Leibleer divergence
        if self.method == 'kl':
            return self._kl(size)

    def _vote_entropy(self, size):
        X_ids, X = self.dataset.get_unlabeled_entries()

        ## Maximum label Number
        c = np.max(self.dataset.unique_labels())

        ## self.dataset.get_num_of_labels() is the number of distinct classes
        votes = np.zeros((len(X), c+1))

        ## Train
        self.teach_committee()

        ## Collect the votes
        for clf in self.committee:
            count = 0
            for sample in X:
                y = clf.predict(np.array([sample]))[0]
                votes[count, y] += 1
                count += 1

        ## Convert to probabilities
        votes = votes/self.n

        ## Convert to entropy
        vote_entropy = np.array([entropy(e) for e in votes])

        ## Zip with entry ids
        zipped_votes = zipit(X_ids, vote_entropy)

        ## Sort it
        result = sort_by_2nd(zipped_votes, 'max')

        return result[:size, 0].astype(int)

    def _kl(self, size):
        X_ids, X = self.dataset.get_unlabeled_entries()

        ## starting array
        proba = np.zeros((1, len(X), self.dataset.get_num_of_labels()))

        ## Train
        self.teach_committee()

        ## Collect the probabilities
        for clf in self.committee:
            proba_of_one_clf = np.zeros(self.dataset.get_num_of_labels())
            for sample in X:
                proba_of_one_clf = np.vstack((proba_of_one_clf, clf.predict_proba(np.array([sample]))))
            proba = np.vstack((proba, np.array([proba_of_one_clf[1:]])))
        proba = proba[1:]

        ## Calculate KL divergence, proba.shape = (n_students, n_samples, n_classes)
        consensus = np.mean(proba, axis = 0) # consensus.shape = (n_samples, n_classes)
        consensus = np.tile(consensus, (self.n, 1, 1)) # consensus.shape = (n_students, n_samples, n_classes)
        kl = np.sum(kl_div(proba, consensus), axis = 2) # kl.shape = (n_students, n_samples)
        kl = np.mean(kl, axis = 0) # kl.shape = (n_samples)

        ## zip it
        zipped_kl = zipit(X_ids, kl)

        ## Sort it
        result = sort_by_2nd(zipped_kl, 'max')

        return result[:size, 0].astype(int)



    def confidence(self):
        pass
