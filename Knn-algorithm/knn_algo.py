import numpy as np
from collections import Counter

#Parent Class
class KnnBase:
    
    def __init__(self, k = 5):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def get_k_nearest_neighbors(self, x):
        distances = np.sqrt(np.sum((self.x_train - x) ** 2, axis = 1))
        nearest_indicies = np.argsort(distances)[:self.k]
        return self.y_train[nearest_indicies]
    
class KnnClassifier(KnnBase):
    
    def predict(self, X):
        X = np.atleast_2d(X)
        return np.array([self._predict_one_(x) for x in X])
    
    def _predict_one_(self, x):
        labels = self.get_k_nearest_neighbors(x)
        most_common = Counter(labels).most_common(1)
        return most_common[0][0]

class KnnRegression(KnnBase):
    
    def predict(self, X):
        X = np.atleast_2d(X)
        return np.array([self._predict_one_(x) for x in X])
    
    def _predict_one_(self, x):
        values = self.get_k_nearest_neighbors(x)
        return np.mean(values)


    
