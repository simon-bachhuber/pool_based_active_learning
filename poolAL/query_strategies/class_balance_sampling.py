from .core import QueryStrategy, Model, Dataset
import numpy as np
from .core.utils import zipit, sort_by_2nd

class ClassBalanceSampling(QueryStrategy):
    '''
    Query strategy that tries to query samples that are likely to balance the training dataset the best in terms of class ratios.
    '''

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset)

        self.model = kwargs.pop('model', None)

        # Sanity Check
        if not isinstance(self.model, Model):
            raise TypeError('model parameter is not of type Model')

    def make_query(self, size = 1):
        return self._get_scores()[:size, 0].astype(int)

    def confidence(self):
        a = self._get_scores()
        a[:,1] = a[:,1]/np.sum(a[:,1])
        return a

    def _get_scores(self):
        # Train classifier
        self.model.train(self.dataset)

        # Obtain probabilities
        unlabeled_ids, unlabeled_samples = self.dataset.get_unlabeled_entries()
        pred = self.model.predict_proba(unlabeled_samples)

        # Obtain number of samples per class
        n_class = list(self.dataset.class_balance().values())
        N = self.dataset.len_labeled()

        # Calculate the score of the form
        # P = (1-n1/N) p_n1 + (1-n2/N) p_n2
        # in binary case
        temp = np.zeros(len(unlabeled_samples))
        count = -1

        for p in pred:
            count += 1
            for l in range(len(n_class)):
                temp[count] += (1- n_class[l]/N)*p[l]

        # Zip and sort it
        scores = sort_by_2nd(zipit(unlabeled_ids, temp), 'max')

        return scores
