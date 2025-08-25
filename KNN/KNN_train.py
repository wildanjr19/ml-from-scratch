import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from modules import KNN

# datasets
iris = datasets.load_iris()

# x dan y
X, y = iris.data, iris.target

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = KNN(k=3)

# training
clf.fit(X_train, y_train)

# predictions
predictions = clf.predict(X_test)

# evaluasi (akurasi)
acc = np.sum(predictions == y_test) / len(y_test)
print(f"Akurasi: {acc}")