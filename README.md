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
us = UncertaintySampling(data, model = clf)

# Obtain entry id of sample to label next
idx = us.make_query()[0] 


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
# Plot of average of 200 runs below

```

<p align="center">
<img src="https://github.com/SimiPixel/pool_based_active_learning/blob/master/readme_plot.svg" width="650">
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
from query_strategies.core import Model 
# or 
from query_strategies.core.models.model import Model

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
