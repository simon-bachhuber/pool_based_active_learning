from .core import QueryStrategy, Dataset, Model
from .uncertainty_sampling import UncertaintySampling
from sklearn.cluster import KMeans
import numpy as np
from .core.utils import zipit, sort_by_2nd, unzipit
from sklearn.metrics.pairwise import cosine_similarity

class DensityWeightedUncertaintySampling(QueryStrategy):
    '''
    Query strategy that weights the US score by a density.

    Algorithm:
    - Use Kmeans to find cluster centers
    - Use Cosine-Similarity to calculate the similarity between sample
      and its corresponding cluster center
    - The score of a sample is its UncertaintySampling score times its similarity score
    - Chose the sample with the highest score to be queried

    Parameters
    ----------

    dataset: {poolAL.query_strategy.core.Dataset}
        Should contain every possible class label at least once.

    model: {poolAL.query_strategy.core.Model}
        The classifier for calculating class probabilities

    method: {string} either 'entropy', 'margin' or 'lc'
        default = 'entropy'

    beta: {int}
        How strong the similarity measure weights the us scores.
        beta = 1 means they are weighted linearly
        default = 1


    Methods
    -------

    .make_query(size = 1)


     References
    ----------
    .. [1] Settles, Burr. "Active learning literature survey." University of
           Wisconsin, Madison 52.55-66 (2010): 11.

    '''

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset)

        self.model = kwargs.pop('model', None)

        self.beta = kwargs.pop('beta', 1)

        if not isinstance(self.model, Model):
            raise TypeError(
            'Parameter model is not an object of type query_strategy.core.Model'
            )

        # UncertaintySampling
        self.US = UncertaintySampling(self.dataset, model = self.model, method = kwargs.pop('method', 'entropy'))

        # Wether us_scores are to be maximed or minimized
        if hasattr(self.model, 'predict_proba') and (self.US.method in ['entropy', 'lc']):
            self.max_min = 'max'
        else:
            self.max_min = 'min'

        # Clustering
        self.KMeans = KMeans(n_clusters=self.dataset.get_num_of_labels())
        self.similarity = None
        self.l = None
        self._clustering()


    def _clustering(self):

        X, _ = self.dataset.get_entries()

        # Only recluster if dataset changed / if you appended a sample
        if len(X) != self.l:
            _, X_pool = self.dataset.get_unlabeled_entries()

            # Update the wether to recluster parameter
            self.l = len(X)

            # (Re)fit the KMeans
            self.KMeans.fit(X)

            # Predict the cluster labels
            cluster_label = self.KMeans.predict(X_pool)

            # Save cluster center
            cluster_center = self.KMeans.cluster_centers_

            self.similarity = []
            for i in range(len(X_pool)):
                self.similarity.append(cosine_similarity(
                    X_pool[i].reshape(1,-1), cluster_center[cluster_label[i]].reshape(1,-1)
                )[0,0])

            self.similarity = np.asarray(self.similarity)

            if self.max_min == 'min':
                self.similarity = 1-self.similarity

    def make_query(self, size = 1):

        # Get uncertainty scores
        ids, us_scores = unzipit(self.US._get_scores())

        # Recluster
        self._clustering()

        scores = us_scores * self.similarity**self.beta
        aux_deleter = zipit(np.arange(len(scores)), scores)
        scores = zipit(ids, scores)

        # Sort them
        aux_deleter = sort_by_2nd(aux_deleter, self.max_min)
        scores = sort_by_2nd(scores, self.max_min)

        # Delete the chosen ones out of self.similarity
        self.similarity = np.delete(self.similarity, aux_deleter[:size, 0].astype(int))

        return scores[:size, 0].astype(int)

    def confidence(self):
        pass
