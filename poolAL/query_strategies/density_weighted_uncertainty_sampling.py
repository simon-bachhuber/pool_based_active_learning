from .core import QueryStrategy, Dataset, Model
from .uncertainty_sampling import UncertaintySampling
from sklearn.cluster import KMeans
import numpy as np
from .core.utils import zipit, sort_by_2nd, unzipit
from sklearn.metrics.pairwise import pairwise_kernels as pw

def normalized_pw(x,y, **kwargs):
    _x = np.sqrt(pw(x, x, **kwargs))
    _y = np.sqrt(pw(y, y, **kwargs))
    return pw(x, y, **kwargs)/(_x*_y)

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

    kernel: {string} out of [‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’, ‘laplacian’, ‘sigmoid’, ‘cosine’]
        Metric to determine the similarity between two samples, should output value between 0 (not similiar) and 1 (equal).
        Underlying function is sklearn.metrics.pairwise.pairwise_kernels.
        default = 'linear'


    Methods
    -------

    .make_query(size = 1)


     References
    ----------
    .. [1] Settles, Burr. "Active learning literature survey." University of
           Wisconsin, Madison 52.55-66 (2010): 11. Page 25.

    '''

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset)

        self.model = kwargs.pop('model', None)

        self.beta = kwargs.pop('beta', 1)

        # Pairwise metric function
        self.kernel = kwargs.pop('kernel', 'linear')

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

        # Aux deleter : Save which samples were queried so that you can delete them from similarity
        #               in next iteration
        self.aux_deleter = None

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
                self.similarity.append(normalized_pw(
                    X_pool[i].reshape(1,-1), cluster_center[cluster_label[i]].reshape(1,-1),
                    metric = self.kernel
                )[0,0])

            self.similarity = np.asarray(self.similarity)

            if self.max_min == 'min':
                self.similarity = 1-self.similarity

    def make_query(self, size = 1):

        # Delete samples from similarity from last iteration
        if self.aux_deleter is not None and self.dataset.modified == True:
            self.similarity = np.delete(self.similarity, self.aux_deleter)

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
        self.aux_deleter = aux_deleter[:size, 0].astype(int)
        self.dataset.modified = False

        return scores[:size, 0].astype(int)

    def confidence(self):
        pass
