import numpy as np
from sklearn import datasets
from modules import SVM
from sklearn.model_selection import train_test_split

# dataset
X, y = datasets.make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.05, random_state=40)
y = np.where(y == 0, -1, 1)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

def hinge_loss(y_true, y_pred):
    loss = np.maximum(0, 1 - y_true * y_pred)
    return np.mean(loss)

clf = SVM()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
# evaluasi
print("Akurasi :", hinge_loss(y_test, predictions))