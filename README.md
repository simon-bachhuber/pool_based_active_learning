# pool_based_active_learning

## Available Query strategies
- QueryStrategy objects
  - RandomSampling
  - UncertaintySampling
  - ClusterMarginSampling
  - ExpectedErrorReduction
  - MeanDistanceSampling
  - QueryByCommittee
  - RepresentativeSampling
- QueryStrategy object that uses a committee of QueryStrategy objects 
  - RankSampling
  - ActiveLearningByLearning
  - DynamicEnsembleActiveLearning
  
## General Usage
```python
# Assume current file is main.ipynb, for example

# Instantiate Dataset object
# X is np.array of shape (n_samples, n_features), y is list
# Samples with no label (yet) have None as label
from query_strategies.core import Dataset
data = Dataset(X, y)

# Instantiate classifier
from query_strategies.core import SVM
clf = SVM()

# Instantiate query strategy
from query_strategies import UncertaintySampling
us = UncertaintySampling(data, clf)

# Obtain entry id of sample to label next
idx = us.make_query()

# Add label to data
data.update(idx, label)

```

## Comparing query strategies
```python
# Import Iris dataset and shuffle it
from sklearn.datasets import load_iris
from sklearn.utils import shuffle

X, y = load_iris(return_X_y = True)
y = y.tolist()
X, y = shuffle(X, y)

# Declare Dataset objects
from query_strategies.core import Dataset
data_train1 = Dataset(X[:100], y[:5]+95*[None])
data_train2 = Dataset(X[:100], y[:5]+95*[None])
data_test = Dataset(X[100:], y[100:])

# Instantiate classifier
from query_strategies.core import SVM
clf = SVM(kernel = 'linear', gamma = 15, random_state = 1)

# Instantiate query strategies
from query_strategies import RandomSampling, UncertaintySampling
rs = RandomSampling(data_train1)
us = UncertaintySampling(data_train2)

# Make queries and update data
idx = rs.make_query()
data_train1.update(idx, y[idx])
idx = us.make_query()
data_train2.update(idx, y[idx])

# Train and score
clf.train(data_train1)
score1 = clf.score(data_test)
clf.train(data_train2)
score2 = clf.score(data_test)

# Iterate until all samples are labeled and plot of the mean of several runs below

```
![Plot of performance](https://github.com/SimiPixel/pool_based_active_learning/blob/master/readme_plot.svg)
  
