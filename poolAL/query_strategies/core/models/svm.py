from .model import Model
from sklearn.svm import SVC


class SVM(Model):
    '''
    SVM with default probability = 1
    '''
    def __init__(self, *args, **kwargs):
        self.model = SVC(probability = kwargs.pop('probability', True), *args, **kwargs)
        supports_prob = hasattr(self.model, 'predict_proba')

        if supports_prob:
            self.__class__ = Aux

    def train(self, dataset):
        X, y = dataset.get_labeled_entries()
        self.model.fit(X, y)

    def decision_function(self, feature):
        return self.model.decision_function(feature)

    def predict(self, feature):
        return self.model.predict(feature)

    def score(self, dataset):
        X, y = dataset.get_labeled_entries()
        return self.model.score(X, y)

class Aux(SVM):
    def predict_proba(self, feature):
        return self.model.predict_proba(feature)
