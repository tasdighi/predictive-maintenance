from sklearn.ensemble import IsolationForest

class Model:
    def __init__(self, contamination=0.02, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
    

    def train_isolation_forest(self, X):
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.model.fit(X)
        return self.model