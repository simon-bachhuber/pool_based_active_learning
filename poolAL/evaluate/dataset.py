#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
The dataset class used.
'''

import numpy as np

class Dataset(object):
    '''
    Dataset object.

    Parameters
    ----------
    X: {np array}, shape = (n_samples, n_features)

    y: list of {int, None}, shape = (n_samples)


    Attributes
    ----------
    data: list, shape = (n_samples)
        List of all sample features and label tuple.

    '''

    def __init__(self, X=None, y=None):
        if X is None: X = np.array([])
        if y is None: y = []

        y = np.array(y)

        self._X = X
        self._y = y
        self.modified = True

    def __len__(self):
        '''
        Number of all sample entries in object.

        Returns
        -------
        n_samples: int
        '''
        return self._X.shape[0]

    def __getitem__(self, idx):
        # interface to direct access the data by index
        return self._X[idx], self._y[idx]

    def delete_entry(self, idx):
        self._X = np.delete(self._X, (idx), 0)
        self._y = np.delete(self._y, (idx), 0)

    @property
    def data(self):
        return self

    def get_labeled_mask(self):
        '''
        Obtain mask of labeled entries.

        Returns
        -------
        mask: np array of bool, shape = (n_samples,)
        '''
        return ~np.fromiter((e is None for e in self._y), dtype = bool)

    def get_mask_of_label(self, k):
        '''
        Obtain mask of labeled entries with label k

        Returns:
        -------
        mask: np.array of bool, shape = (n_samples,)
        '''
        return np.fromiter((e is k for e in self._y), dtype = bool)

    def len_labeled(self):
        '''
        Number of labeled data in object.

        Returns:
        --------
        n_samples: int

        '''
        return self.get_labeled_mask().sum()

    def len_unlabeled(self):
        '''
        Number of unlabeled data in object.

        Returns:
        --------
        n_samples: int
        '''
        return (~self.get_labeled_mask()).sum()

    def get_num_of_labels(self):
        '''
        Number of distinct labels in object.

        Returns:
        --------
        n_labels: int
        '''
        return np.unique(self._y[self.get_labeled_mask()]).size

    def append(self, feature, label = None):
        '''
        Add a (feature, label) entry to this object.
        A None label indicates an unlabeled entry.

        Parameters
        ----------
        feature: {array-like}, shape = (n_features)
            Feature of the sample to append.

        label: {int, None}
            Label of the sample.

        Returns:
        --------
        entry_id: {int}
            entry_id of the appended sample.
        '''
        self._X = np.vstack((self._X, feature))
        self._y = np.append(self._y, label)

        self.modified = True
        return len(self) - 1

    def update(self, entry_id, new_label):
        '''
        Updates an entry with a new label.

        Parameters
        ----------
        entry_id: int
            entry id of the sample to update.

        new_label: {int, None}
            label of the sample to update.
        '''
        self._y[entry_id] = new_label
        self.modified = True

    def get_entries(self):
        '''
        Return tuple of all samples features and labels

        Returns
        -------
        X: np.array, shape = (n_samples, n_features)
        y: np.array, shape = (n_samples,)
        '''
        return self._X, self._y

    def get_labeled_entries(self):
        '''
        Return tuple of all labeled samples features and labels

        Returns:
        --------
        X: np.array
        y: list
        '''
        mask = self.get_labeled_mask()
        return self._X[mask], (self._y[mask]).tolist()

    def get_labeled_entries_ids(self):
        '''
        Return entry ids of all labeled samples

        Returns:
        --------
        np.arrary, shape = (n_labeled_samples)
        '''
        mask = self.get_labeled_mask()
        return (np.where(mask)[0]).astype(int)

    def get_unlabeled_entries(self):
        '''
        Return tuple of all unlabeled samples entry_ids and features

        Returns:
        --------
        entry_id: np.array
        X: np.array
        '''
        mask = ~self.get_labeled_mask()
        return (np.where(mask)[0]).astype(int), self._X[mask]

    def labeled_uniform_sample(self, sample_size, replace = True):
        '''
        Returns a Dataset object with labeled data only,
        which is resampled uniformly with given sample size.

        Parameters
        ----------
        samples_size: int
        replace: bool, wether or not to sample with replacement
        '''
        # Set of unique labels
        unique_labels = list(set(self._y))

        # First ensure that every class is present at least once
        idx = np.zeros(1).astype(int)
        for l in unique_labels:
            idx = np.hstack((idx, np.random.choice(np.where(self.get_mask_of_label(l))[0], size = 1)))

        if sample_size > len(unique_labels):
            idx = np.hstack((idx, np.random.choice(np.setdiff1d(np.where(self.get_labeled_mask())[0], idx),
                                  size = sample_size-len(unique_labels), replace = replace)))
        idx = idx[1:]

        return Dataset(self._X[idx], self._y[idx])



# In[ ]:
