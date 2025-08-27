import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from modules import LogisticRegression

# load data
df = datasets.load_breast_cancer()
X, y = df.data, df.target

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# evaluasi
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# buat instance
clf = LogisticRegression(lr=0.0001, n_iters=1000)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# evaluasi
print("Akurasi :", accuracy(y_test, predictions))