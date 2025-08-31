import numpy as np

class LinearRegression:

    def __init__(self, lr: float = 0.0001, n_iters: int = 1000):
        """
        Args:
            lr : learning rate
            n_iters : banyaknya iterasi
            weights : w -> bobot
            bias : b -> bias
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # inisialisasi parameter
        n_samples, n_features = X.shape
        # inisialisasi weights dengan 0 (matriks) dan bias 0 (skalar)
        self.weights = np.zeros(n_features)
        self.bias = 0

        # iterasi sebanyak n_iters
        for _ in range(self.n_iters):
            # hitung prediksi awal sebelum pelatihan
            y_predicted = np.dot(X, self.weights) + self.bias

            # hitung gradien dw dan db
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update weight dan bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted