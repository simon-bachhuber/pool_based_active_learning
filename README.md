# pool_based_active_learning

## Installation
Run in Terminal
```
pip install git+https://github.com/SimiPixel/pool_based_active_learning.git
```

## Available Query strategies
- QueryStrategy objects
  - RandomSampling
  - UncertaintySampling
  - DensityWeightedUncertaintySampling
  - ClusterMarginSampling
  - ClassBalanceSampling
  - ExpectedErrorReduction
  - MeanDistanceSampling
  - QueryByCommittee
  - RepresentativeSampling
  - NearestNeighbourCriterion
- QueryStrategy object that uses a committee of QueryStrategy objects
  - RankSampling
  - ActiveLearningByLearning
  - DynamicEnsembleActiveLearning

## General Usage
```python

# Instantiate Dataset object
# X is np.array of shape (n_samples, n_features), y is list
# Samples with no label (yet) have None as label
from poolAL.query_strategies.core import Dataset
data = Dataset(X, y)

# Instantiate classifier
from poolAL.query_strategies.core import SVM
clf = SVM()

# Instantiate query strategy
from poolAL.query_strategies import UncertaintySampling
us = UncertaintySampling(data, model = clf)

# Obtain entry id of sample to label next
idx = us.make_query()[0]

# Add label to data
data.update(idx, label)

```

## Comparing query strategies
### Manually
```python
# Import Iris dataset and shuffle it
from sklearn.datasets import load_iris
from from poolAL.query_strategies.core.utils import shuffle

X, y = load_iris(return_X_y = True)
y = y.tolist()
X, y = shuffle(X, y, 3)

# Declare Dataset objects
from poolAL.query_strategies.core import Dataset
data_train1 = Dataset(X[:100], y[:5]+95*[None])
data_train2 = Dataset(X[:100], y[:5]+95*[None])
data_test = Dataset(X[100:], y[100:])

# Instantiate classifier
from poolAL.query_strategies.core import SVM
clf = SVM(kernel = 'linear', random_state = 1)

# Instantiate query strategies
from poolAL.query_strategies import RandomSampling, UncertaintySampling
rs = RandomSampling(data_train1)
us = UncertaintySampling(data_train2, model = clf)

# Make queries and update data
idx = rs.make_query()[0]
data_train1.update(idx, y[idx])
idx = us.make_query()[0]
data_train2.update(idx, y[idx])

# Train and score
clf.train(data_train1)
score1 = clf.score(data_test)
clf.train(data_train2)
score2 = clf.score(data_test)

# Iterate until all samples are labeled and take the mean of several runs

```
### Automated
```python
from poolAL.evaluate import CalcScore, CalcScoreParallel
from poolAL.query_strategies import RandomSampling, ClusterMarginSampling, UncertaintySampling
from poolAL.query_strategies.core import SVM

# Instantiate classifier
clf = SVM(kernel = 'linear')

# Classifier kwargs as a dict
clf_kwargs = {'kernel': 'linear'}

# Declare query strategies and their respective keyword arguments
qs = [RandomSampling, ClusterMarginSampling, UncertaintySampling]
qs_kwargs = [{}, {'space': 'full'}, {'model': clf}]

# Load Iris
X, y = load_iris(return_X_y = True)
y = y.tolist()

# Number of labels to start with
n_labels_start = 3
# Number of labels to end
n_labels_end = 100
# Number of classes present in initial labels
n_unique_labels = 3
# Number of runs for averaging
n_runs = 200

test_scores = CalcScore(X, y, qs, qs_kwargs, SVM, clf_kwargs, n_labels_start, n_labels_end, n_runs, n_unique_labels)

# CalcScoreParallel instead of CalcScore for a cpu-parallized version
# Plot of test_scores below

```

<p align="center">
<img src="https://github.com/SimiPixel/pool_based_active_learning/blob/master/readme_plot.svg" width="650">
</p>

## Visualisation of UncertaintySampling: Why it works
```python
from sklearn.datasets import load_iris
from poolAL.query_strategies.core.utils import shuffle

# load again Iris and shuffle
X, y = load_iris(return_X_y = True)
y = y.tolist()
X, y = shuffle(X, y, 3)

from poolAL.query_strategies.core import Dataset, SVM

# Declare Dataset and classifier
d = Dataset(X, y[:100]+50*[None])
clf = SVM(kernel = 'linear')

from poolAL.query_strategies import UncertaintySampling

# Declare query strategy
US = UncertaintySampling(d, model = clf)

from poolAL.visualize import Visualizer

vis = Visualizer(US, y=y, conf_gradient = True)
vis.next()
vis.plot()

```
<p align="center">
<img src="https://github.com/SimiPixel/pool_based_active_learning/blob/master/readme_plot2.svg" width="1100">
</p>

## Using active learning on your own classifier
Some QueryStrategies require a classifier to base their query desicion on, e.g., UncertaintySampling queries the samples that a given classifier is most uncertain off. This classifier must be a Model object.
There are two different classifiers already built in inside the folder query_strategies.core.models:
- SVM (from query_strategies.core import SVM)
- RandomForestClassifer (from query_strategies.core import RFC)

But what if we want to use a Multi-layer Perceptron classifier ..
```python
# Import the classifier
from sklearn.neural_network import MLPClassifier

# Import the Model class
from poolAL.query_strategies.core import Model
# or
from poolAL.query_strategies.core.models.model import Model

# Declare the Model
class MLP(Model):
    def __init__(self, *args, **kwargs):
        self.model = MLPClassifier(*args, **kwargs)

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

```

## Citing
```
@techreport{YY2017,
  author = {Yao-Yuan Yang and Shao-Chuan Lee and Yu-An Chung and Tung-En Wu and Si-An Chen and Hsuan-Tien Lin},
  title = {libact: Pool-based Active Learning in Python},
  institution = {National Taiwan University},
  url = {https://github.com/ntucllab/libact},
  note = {available as arXiv preprint \url{https://arxiv.org/abs/1710.00379}},
  month = oct,
  year = 2017
}
```
