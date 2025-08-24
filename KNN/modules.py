import numpy as np
from collections import Counter

# rumus euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# KNN Class
class KNN:

    def __init__(self, k = 3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # X = multiple sampel
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    # helper
    def _predict(self, x):
        # x = one sampel

        # hitung jarak
        distance = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # dapatkan k terdekat
        k_idx = np.argsort(distance)[:self.k]
        k_neares_labels = [self.y_train[i] for i in k_idx]
        # majority vote (common)
        most_common = Counter(k_neares_labels).most_common(1)
        return most_common[0][0]