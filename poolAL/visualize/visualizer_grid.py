import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import copy
import matplotlib.pyplot as plt
from ..query_strategies.core import QueryStrategy

class VisualizerGrid:
    '''
    Visualizes the process of the query strategy when making a desicion in 2d-data space.

    Parameters:
    -----------

    qs: {poolAL.query_strategy.core.QueryStrategy}
        The query strategy to visualize.

    y: {list}
        List of labels of all samples saved in qs.dataset
        default = None

    classifier: optional

    dim_pca: {int}
        The number of dimension after PCA is used to dimension reduce.
        Then TSNE will always reduce further to two dimensions.
        default = 5

    n_grid: {int}
        The grid size per dimension. The number of points in the grid is
        n_grid**dim_pca
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

        self.dim_pca = kwargs.pop('dim_pca', 5)

        # Save labels
        self._y = kwargs.pop('y', None)
        if self._y is None:
            raise TypeError('y parameter is not set')

        # random state for dimension reduction
        self.random_state = kwargs.pop('random_state', None)

        # Save samples
        self._X = self.qs.dataset._X

        # PCA
        self.pca = PCA(n_components=self.dim_pca, random_state=self.random_state)
        self._X = self.pca.fit_transform(self._X)

        # Replace samples by pca reduced ones
        self.qs.dataset._X = self._X

        # save dataset
        self.dataset = self.qs.dataset

        # TSNE
        self.tsne = TSNE(perplexity = kwargs.pop('perplexity', 30), random_state=self.random_state)

        # Labeled entries
        self.labeled_ids = self.dataset.get_labeled_entries_ids()

        # Next to query
        self.query_ids = None

        # Class color table
        self.color = ['black', 'red', 'blue', 'green', 'yellow', 'orange', 'brown', 'purple', 'pink','gray', 'white']

        # Assign each class label its color
        self._y_color = self._assign_color(self._y)

        # Grid size per dimension
        self.n_grid = kwargs.pop('n_grid', 5)

        # Total grid points
        self.size = self.n_grid**self.dim_pca

        # Calculate the grid
        self.grid_X = self._get_grid()

        # Generate query strategy with grid points
        self.qs_grid = copy.deepcopy(self.qs)
        self.qs_grid.dataset._X = np.vstack((self._X, self.grid_X))
        self.qs_grid.dataset._y = np.concatenate((self.qs_grid.dataset._y, np.array([None for _ in range(self.size)])))

        # Transform to embedded space
        self.embedded_X = self.tsne.fit_transform(self.qs_grid.dataset._X)
        self.embedded_X_grid = self.embedded_X[-self.size:]
        l = len(self.embedded_X)
        self.embedded_X_samples = self.embedded_X[:(l-self.size)]

        # Classifier if query strategy doesent have one
        self.clf = kwargs.pop('classifier', self.qs.model)

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

    def _get_grid(self):

        mi = np.min(self._X, axis = 0)
        ma = np.max(self._X, axis = 0)

        coor = []
        for i in range(self.dim_pca):
            coor.append(np.linspace(mi[i], ma[i], self.n_grid))

        mesh = np.meshgrid(*coor)

        # Flatten the meshgrid
        for i in range(self.dim_pca):
            mesh[i] = mesh[i].flatten()

        # Convert meshgrid into form (n_samples, n_features)
        a = copy.copy(mesh[0].reshape((mesh[0].shape+(1,))))
        a.fill('nan')

        for arr in mesh:
            a = np.concatenate((a, arr.reshape((arr.shape+(1,)))), axis = -1)

        return np.apply_along_axis(lambda a: a[1:], axis = -1, arr = a)

    def next(self):
        '''
        Adds one label.
        '''

        # Add one label
        if self.query_ids is not None:
            id = self.query_ids[0]
            self.dataset.update(id, self._y[id])
            self.qs_grid.dataset.update(id, self._y[id])

        # Do next query
        self.query_ids = self.qs.make_query()

        # Update labeled ids
        self.labeled_ids = self.dataset.get_labeled_entries_ids()

        # Update predictions
        self.pred = self.clf.predict(self.qs_grid.dataset._X)
        # Convert to colors
        self.pred = self._assign_color(self.pred)

        # Calculate confidences
        self.conf = self.qs_grid.confidence()#[-self.size:]


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

        fig, ax = plt.subplots(3,1, figsize=(10,25))

        ##### SUBPLOT 1

        # Mark the labeled ones
        ax[0].scatter(self.embedded_X_samples[self.labeled_ids,0], self.embedded_X_samples[self.labeled_ids,1],
         edgecolors='deepskyblue', c='None', s=200)

        if draw_class_labels:
            # Draw all
            ax[0].scatter(self.embedded_X_samples[:,0], self.embedded_X_samples[:,1], c=self._y_color, edgecolors='black', s=40)
        else:
            ax[0].scatter(self.embedded_X_samples[:,0], self.embedded_X_samples[:,1], c='white', edgecolors='black', s=40)

        xlim, ylim = ax[0].get_xlim(), ax[0].get_ylim()

        data = self.embedded_X
        ##### SUBPLOT 2: predictions

        if self.pred is not None:
            ax[1].scatter(self.embedded_X_samples[self.labeled_ids,0], self.embedded_X_samples[self.labeled_ids,1],
             edgecolors='deepskyblue', c='None', s=200)
            ax[1].scatter(data[:,0], data[:,1], c=self.pred, s=40)

        ##### SUBPLOT 3: confidence

        mask = ~self.qs_grid.dataset.get_labeled_mask()

        if self.conf is not None:
            # Color gradient
            ax[2].scatter(self.embedded_X_samples[self.labeled_ids,0], self.embedded_X_samples[self.labeled_ids,1],
             edgecolors='deepskyblue', c='None', s=200)
            c = ax[2].scatter(data[mask,0], data[mask,1], c=self.conf, s=40, cmap=kwargs.pop('cmap', 'winter'))

            plt.colorbar(c)

        # Set the axis limits
        for i in [1,2]:
            ax[i].set_xlim(xlim)
            ax[i].set_ylim(ylim)

        # return fig
