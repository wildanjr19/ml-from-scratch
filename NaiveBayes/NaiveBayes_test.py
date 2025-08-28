import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets


from modules import NaiveBayes

# evaluasi
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

clf = NaiveBayes()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print("Akurasi :", accuracy(y_test, predictions))