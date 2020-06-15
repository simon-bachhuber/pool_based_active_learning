import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import copy
import matplotlib.pyplot as plt
from ..query_strategies.core import QueryStrategy
from ..query_strategies.core.utils import get_grid

class VisualizerGrid:
    '''
    Visualizes the process of the query strategy when making a desicion in 2d-data space.
    If dim_pca and dim_svd are both larger than 2 or None, then uses TSNE to always transform to 2 dimensions for visualisation.

    Parameters:
    -----------

    qs: {poolAL.query_strategy.core.QueryStrategy}
        The query strategy to visualize.

    y: {list}
        List of labels of all samples saved in qs.dataset
        default = None

    classifier: optional

    dim_svd: {int} or None
        The number of dimension after PCA is used to dimension reduce.
        If None only uses PCA.
        default = 5

    dim_pca: {int} or None
        The number of dimension after PCA is used to dimension reduce.
        If None only uses SVD
        default = None

    n_grid: {int}
        The grid size per dimension. The number of points in the grid is
        n_grid** Min(dim_pca, dim_svd)
        default = 5

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

        self.dim_pca = kwargs.pop('dim_pca', None)

        self.dim_svd = kwargs.pop('dim_svd', 5)

        #if self.dim_pca is None and self.dim_svd is None:
        #    raise ValueError('Both dim_svd and dim_pca can not be None. At least one must be given.')

        if self.dim_pca == self.dim_svd and self.dim_pca is not None:
            raise ValueError('dim_svd and dim_pca should not be equal.')

        self.both = False
        if self.dim_pca is not None and self.dim_svd is not None:
            self.both = True
            self.smaller_dim = min(self.dim_pca, self.dim_svd)
        elif self.dim_pca is None and self.dim_svd is None:
            self.smaller_dim = self.qs.dataset._X.shape[1]
        elif self.dim_pca is None:
            self.smaller_dim = self.dim_svd
        elif self.dim_svd is None:
            self.smaller_dim = self.dim_pca
        else:
            pass

        # Save labels
        self._y = kwargs.pop('y', None)
        if self._y is None:
            raise TypeError('y parameter is not set')

        # random state for dimension reduction
        self.random_state = kwargs.pop('random_state', None)

        # Save samples
        self._X = self.qs.dataset._X

        # PCA/SVD
        # Do both
        if self.both:
            self.pca = PCA(n_components=self.dim_pca, random_state=self.random_state)
            self.svd = TruncatedSVD(n_components=self.dim_svd, random_state=self.random_state)

            # First PCA
            if self.dim_pca > self.dim_svd:
                self._X = self.pca.fit_transform(self._X)
                self._X = self.svd.fit_transform(self._X)
            # First SVD
            else:
                self._X = self.svd.fit_transform(self._X)
                self._X = self.pca.fit_transform(self._X)
        # None for both, do nothing
        elif self.dim_pca is None and self.dim_svd is None:
            pass
        # Only SVD
        elif self.dim_pca is None:
            self.svd = TruncatedSVD(n_components=self.dim_svd, random_state=self.random_state)
            self._X = self.svd.fit_transform(self._X)
        # Only PCA
        elif self.dim_svd is None:
            self.pca = PCA(n_components=self.dim_pca, random_state=self.random_state)
            self._X = self.pca.fit_transform(self._X)
        else:
            raise Exception('Something went wrong ..')

        # Replace samples by reduced ones
        self.qs.dataset._X = self._X
        self.qs.dataset.modified_X = True

        # save dataset
        self.dataset = self.qs.dataset

        # TSNE
        self.tsne = TSNE(perplexity = kwargs.pop('perplexity', 30), random_state=self.random_state)

        # Labeled entries
        self.labeled_ids = self.dataset.get_labeled_entries_ids()

        # Next to query
        self.query_ids = None

        # Class color, marker table
        self.color = ['orange', 'deepskyblue', 'limegreen', 'red', 'gray', 'yellow', 'black', 'brown', 'purple', 'pink', 'green', 'blue', 'white']
        self.marker = ['o', '^', '1', 'X', 'h', 'D', None, None, None, None, None, None, None]

        # Assign each class label its color and marker
        self._y_color = self._assign_color(self._y)

        # Grid size per dimension
        self.n_grid = kwargs.pop('n_grid', 5)

        # Calculate the grid
        self.grid_X = get_grid(self._X, self.n_grid)
        self.full_X = np.vstack((self._X, self.grid_X))

        # Total number of grid points
        self.size = self.n_grid**self.smaller_dim

        # Check
        if self.size != len(self.grid_X):
            raise Exception('There is something going wrong when computing the grid')

        # Transform to embedded space
        # But only if required, so if the dimension is still larger than 2
        if self.smaller_dim > 2:
            self.embedded_X = self.tsne.fit_transform(self.full_X)
        else:
            self.embedded_X = self.full_X

        self.embedded_X_grid = self.embedded_X[-self.size:]
        l = len(self.embedded_X)
        self.embedded_X_samples = self.embedded_X[:(l-self.size)]

        # Classifier if query strategy doesent have one
        self.clf = kwargs.pop('classifier', None)
        if self.clf is None:
            self.clf = self.qs.model

        # predictions
        self.pred = None

        # confidences
        self.conf = None


    def _assign_color(self, y):

        c = []
        label_names = np.unique(self._y)

        for i in range(len(y)):
            for j in range(len(label_names)):
                if y[i] == label_names[j]:
                    c.append(self.color[j])
                    break

        return c


    def next(self):
        '''
        Adds one label.
        '''

        # Add one label
        if self.query_ids is not None:
            id = self.query_ids[0]
            self.dataset.update(id, self._y[id])

        # Do next query
        self.query_ids = self.qs.make_query()

        # Update labeled ids
        self.labeled_ids = self.dataset.get_labeled_entries_ids()

        # Fit if classifier is not fitted in .make_query
        if not hasattr(self.qs, 'model'):
            self.clf.train(self.dataset)

        # Update predictions
        self.pred = self.clf.predict(self.grid_X)
        # Convert to colors
        self.pred = self._assign_color(self.pred)

        # Calculate confidences
        self.conf = self.qs.confidence_grid(self.grid_X)


    def plot(self, draw_class_labels = True, **kwargs):
        '''
        Plot one frame.

        Parameters:
        -----------

        cmap: {string}

        draw_class_labels: {bool}
            If True, every points color corresponds to its class membership.
            Otherwise all points are white with black edge.
            default = True

        draw_colorbar: {bool}
            default = False

        '''

        fig, ax = plt.subplots(1,3, figsize=(35,10))

        # Set ticks
        size = kwargs.pop('tick_size', 12)
        xticks = kwargs.pop('xticks', [])
        yticks = kwargs.pop('yticks', [])

        for i in range(3):
            ax[i].tick_params(labelsize = size)
            ax[i].set_xticks(xticks)
            ax[i].set_yticks(yticks)

        ##### SUBPLOT 1

        # Mark the labeled ones
        ax[0].scatter(self.embedded_X_samples[self.labeled_ids,0], self.embedded_X_samples[self.labeled_ids,1],
         edgecolors='None', c='lime', s=1000)

        # Mark the next in line
        if self.query_ids is not None:
            idx = self.query_ids[0]
            ax[0].scatter(self.embedded_X_samples[idx,0], self.embedded_X_samples[idx, 1], c= 'red', s=1000)

        if draw_class_labels:
            unique_l = np.unique(self._y)
            for i in range(len(np.unique(self._y))):
                mask = np.fromiter([x == unique_l[i] for x in self._y], dtype=bool)
            # Draw all
                ax[0].scatter(self.embedded_X_samples[mask,0], self.embedded_X_samples[mask,1], c=self.color[i],
                 marker = self.marker[i], edgecolors='black', s=250)
        else:
            ax[0].scatter(self.embedded_X_samples[:,0], self.embedded_X_samples[:,1], c='white', edgecolors='black', s=250)

        xlim, ylim = ax[0].get_xlim(), ax[0].get_ylim()

        ##### SUBPLOT 2: predictions

        if self.pred is not None:
            #ax[1].scatter(self.embedded_X_samples[self.labeled_ids,0], self.embedded_X_samples[self.labeled_ids,1],
             #edgecolors='deepskyblue', c='None', s=200)
            ax[1].scatter(self.embedded_X_grid[:,0], self.embedded_X_grid[:,1], c=self.pred, s=40)

        ##### SUBPLOT 3: confidence

        if self.conf is not None:
            # Color gradient
            #ax[2].scatter(self.embedded_X_samples[self.labeled_ids,0], self.embedded_X_samples[self.labeled_ids,1],
             #edgecolors='deepskyblue', c='None', s=200)
            c = ax[2].scatter(self.embedded_X_grid[:,0], self.embedded_X_grid[:,1], c=self.conf, s=40, cmap=kwargs.pop('cmap', 'winter'))

            if kwargs.pop('draw_colorbar', False):
                plt.colorbar(c)

        # Set the axis limits
        for i in [1,2]:
            ax[i].set_xlim(xlim)
            ax[i].set_ylim(ylim)



        return fig
