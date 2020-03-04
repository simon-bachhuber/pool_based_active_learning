from .model import Model
from sklearn.ensemble import RandomForestClassifier

class RFC(Model):
    
    def __init__(self, *args, **kwargs):
        self.model = RandomForestClassifier(*args, **kwargs)
        
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