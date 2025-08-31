import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from modules import LinearRegression

# data
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


regressor = LinearRegression(lr=0.01)
regressor.fit(X_train, y_train)

predicted = regressor.predict(X_test)

# MSE
def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)

print("MSE:", mse(y_test, predicted))