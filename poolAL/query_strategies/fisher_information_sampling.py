import numpy as np
from .core import QueryStrategy, Model, Dataset
from .core.utils import zipit, sort_by_2nd

class FisherInformationSampling(QueryStrategy):

    '''
    For now only Binary dataset.

    References
    ----------
        [1]   Xuejun Liao, Ya Xue, Lawrence Carin. Logistic Regression with an Auxiliary Data Source. Section 5.
    '''

    def __init__(self, dataset, **kwargs):

        super().__init__(dataset)

        self.model = kwargs.pop('model', None)
        if not isinstance(self.model, Model):
            raise Exception('model  parameter should be of type .core.models.model.Model')

        self.Q = np.zeros((self.dataset._X[0], self.dataset._X[1], self.dataset._X[1])) # shape (n_samples, n_features, n_features)

    def _update_Q_contributions(self):

        # Fit the classifier
        self.model.train(self.dataset)

        X = self.dataset._X
        # Grab probabilities
        pred = self.model.predict_proba(X)

        for i in range(len(X)):
            self.Q[i] = pred[i]*(1-pred[i])*np.einsum('i,j',X[i], X[i])

        return pred

    def _get_scores(self):

        # Update all Q contributions
        pred = self._update_Q_contributions()

        labeled_ids = self.dataset.get_labeled_entries_ids()

        # Q is the sum of all Q contributions from labeled samples
        Q = np.einsum('ijk->jk', self.Q[labeled_ids])

        unlabeled_ids, X_unlabeled = self.dataset.get_unlabeled_entries()

        det = []
        for i in range(len(unlabeled_ids)):
            Q_temp = Q + self.Q[unlabeled_ids[i]]
            det.append(np.linalg.det(Q_temp))

        det = np.array(det)

        # zip
        return zipit(unlabeled_ids, det)

    def make_query(size = 1):

        score = self._get_scores()

        return sort_by_2nd(score, 'max')[:size,0].astype(int)
