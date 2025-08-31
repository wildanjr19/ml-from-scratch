import numpy as np

class SVM:

    def __init__(self, lr:float  = 0.001, lambda_value:float = 0.01, n_iters:int = 1000):
        self.lr = lr
        self.lambda_value = lambda_value
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # buat batas untuk y
        y_ = np.where(y <= 0, -1, 1)

        # inisialisasi 
        n_samples, n_features = X.shape

        # inisialisasi weight dan bias
        self.w = np.zeros(n_features)
        self.b = 0

        # loop pelatihan
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # buat kondisi untuk gradien
                condition = y_[idx] * np.dot(x_i, self.w) - self.b >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_value * self.w)
                    # db = 0
                else:
                    self.w -= self.lr * (2 * self.lambda_value * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        # linear mode
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)