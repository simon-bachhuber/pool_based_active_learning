import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import copy
import matplotlib.pyplot as plt
from ..query_strategies.core import QueryStrategy

class Visualizer:
    '''
    Visualizes the process of the query strategy when making a desicion in 2d-data space.

    Parameters:
    -----------

    qs: {poolAL.query_strategy.core.QueryStrategy}
        The query strategy to visualize.

    y: {list}
        List of labels of all samples saved in qs.dataset
        default = None

    dim_pca: {int}
        The number of dimension after PCA is used to dimension reduce.
        Then TSNE will always reduce further to two dimensions.
        default = 50

    n_best: {int}
        Number of samples to color code that would have rank 1, 2, .., n_best
        in the ranking of the query strategy
        default = 5

    rank_gradient: {bool}
        If True, then instead of marking the n_best samples of the query, marks all
        unlabeled samples according to their rank in the query strategy ranking
        using a color gradient.
        default = False
        
    conf_gradient: {bool}
        If True, then instead of marking the n_best samples of the query, marks all
        unlabeled samples according to their confidence score in the query.
        default = False

    random_state: {bool}
        Random_state for PCA and TSNE.
        default = None


    Methods:
    --------

    .next():
        Adds one label according to qs.

    .plot():
        Plots the current embedded space

    '''
    def __init__(self, qs, **kwargs):

        self.qs = qs

        # Check that qs is a QueryStrategy object
        if not isinstance(self.qs, QueryStrategy):
            raise TypeError('qs parameter must be a QueryStrategy object')

        self.dim_pca = kwargs.pop('dim_pca', 50)

        # Save labels
        self._y = kwargs.pop('y', None)
        if self._y is None:
            raise TypeError('y parameter is not set')

        # Dataset
        self.dataset = self.qs.dataset

        # random state for dimension reduction
        self.random_state = kwargs.pop('random_state', None)

        # Save samples
        self._X = self.dataset._X

        # PCA
        self.pca = PCA(n_components=self.dim_pca, random_state=self.random_state)

        # TSNE
        self.tsne = TSNE(perplexity = kwargs.pop('perplexity', 30), random_state=self.random_state)

        # Transform to embedded space
        self.embedded_X = self._transform()

        # Labeled entries
        self.labeled_ids = self.dataset.get_labeled_entries_ids()

        # Next to query
        self.query_ids = None

        # Number of samples to color code
        self.size = kwargs.pop('n_best', 5)

        # Class color table
        self.color = ['black', 'red', 'blue', 'green', 'yellow', 'orange', 'brown', 'purple', 'pink','gray', 'white']

        # Assign each class label its color
        self._y_color = self._assign_color()

        # rank_gradient
        self.rank_gradient = kwargs.pop('rank_gradient', False)

        if self.rank_gradient:
            self.size = 10**4
            
        # conf_gradient
        self.conf_gradient = kwargs.pop('conf_gradient', False)
        
        self.conf = None
        
        if self.conf_gradient is True and self.rank_gradient is True:
            raise Exception('Can only plot either ranks or confidence scores')


    def _assign_color(self):

        c = []
        label_names = np.unique(self._y)

        for i in range(len(self._y)):
            for j in range(len(label_names)):
                if self._y[i] == label_names[j]:
                    c.append(self.color[j])
                    break

        return c


    def _transform(self):

        # Copy X
        X = copy.deepcopy(self._X)

        # Do PCA
        if X.shape[1] > 50:
            X = self.pca.fit_transform(X)

        # Do TSNE
        X = self.tsne.fit_transform(X)

        return X

    def next(self):
        '''
        Adds one label.
        '''

        # Add one label
        if self.query_ids is not None:
            id = self.query_ids[0]
            self.dataset.update(id, self._y[id])

        # Do next query
        self.query_ids = self.qs.make_query(self.size)
        
        # Update conf scores
        if self.conf_gradient:
            self.conf = self.qs.confidence()
            if self.conf is None:
                raise Exception('Sorry but this query strategy has no legit confidence method (yet)')

        # Update labeled ids
        self.labeled_ids = self.dataset.get_labeled_entries_ids()

    def plot(self, draw_class_labels = True, **kwargs):
        '''
        Plot one frame.

        Parameters:
        -----------

        cmap: {string}

        draw_class_labels: {int}
            If True, every points color corresponds to its class membership.
            Otherwise all points are white with black edge.
            default = True

        '''

        if self.rank_gradient:
            return self._plot_rank_gradient(draw_class_labels, **kwargs)
        elif self.conf_gradient:
            return self._plot_conf_gradient(draw_class_labels, **kwargs)
        else:
            return self._plot(draw_class_labels)


    def _plot(self, draw_class_labels):

        fig = plt.figure(figsize=(15,12))

        # Mark the labeled ones
        plt.scatter(self.embedded_X[self.labeled_ids,0], self.embedded_X[self.labeled_ids,1], edgecolors='deepskyblue',
                    c='None', s=150)

        if self.query_ids is not None:
            # Mark the number one
            plt.scatter(self.embedded_X[self.query_ids[0],0], self.embedded_X[self.query_ids[0],1], c='deepskyblue', s=200)

            # Mark number two, three, ..., self.size
            plt.scatter(self.embedded_X[self.query_ids[1:],0], self.embedded_X[self.query_ids[1:],1], c='limegreen', s=200)

        if draw_class_labels:
            # Draw all
            plt.scatter(self.embedded_X[:,0], self.embedded_X[:,1], c=self._y_color, edgecolors='black', s=40)
        else:
            plt.scatter(self.embedded_X[:,0], self.embedded_X[:,1], c='white', edgecolors='black', s=40)

        return fig


    def _plot_rank_gradient(self, draw_class_labels, **kwargs):

        fig = plt.figure(figsize=(15,12))

        if self.query_ids is not None:
            # Color gradient
            plt.scatter(self.embedded_X[self.query_ids,0], self.embedded_X[self.query_ids,1], c=np.arange(len(self.query_ids)),
            s=150, cmap=kwargs.pop('cmap', 'winter'))

            plt.colorbar()

        if draw_class_labels:
            # Draw all
            plt.scatter(self.embedded_X[:,0], self.embedded_X[:,1], c=self._y_color, edgecolors='black', s=40)
        else:
            plt.scatter(self.embedded_X[:,0], self.embedded_X[:,1], c='white', edgecolors='black', s=40)

        return fig
    
    def _plot_conf_gradient(self, draw_class_labels, **kwargs):

        fig = plt.figure(figsize=(15,12))

        if self.conf is not None:
            # Color gradient
            idx, _ = self.dataset.get_unlabeled_entries()
            plt.scatter(self.embedded_X[idx,0], self.embedded_X[idx,1], c=self.conf,
            s=150, cmap=kwargs.pop('cmap', 'winter'))

            plt.colorbar()

        if draw_class_labels:
            # Draw all
            plt.scatter(self.embedded_X[:,0], self.embedded_X[:,1], c=self._y_color, edgecolors='black', s=40)
        else:
            plt.scatter(self.embedded_X[:,0], self.embedded_X[:,1], c='white', edgecolors='black', s=40)

        return fig
