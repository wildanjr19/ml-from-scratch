import numpy as np
from collections import Counter
# import decision tree sebelumnya
from utils_dt import DecisionTree


# fungsi bootstrapping untuk menghasilkan subset data
def bootstrapping(X, y):
    # inisialisasi jumlah sampel
    n_samples = X.shape[0]
    # pilih data acak dengan pengembalian
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    # hasilkan subset data -> mengambil dari X dan y sesuai dengan idxs
    return X[idxs], y[idxs]

# fungsi counter untuk voting
def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


class RandomForest:
    def __init__(self, n_trees: int = 100, min_samples_split: int = 2, max_depth: int = 100, n_feats = None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        # buat variabel list kosong untuk menyimpan pohon
        self.tree = []

    def fit(self, X, y):
        # buat list kosong untuk menyimpan pohon dalam proses pelatihan
        self.trees = []

        # looping melatih setiap pohon
        for _ in range(self.n_trees):
            # inisialisasi decision tree
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth, n_feats=self.n_feats)
            # latih pohon dengan subset yang berbeda-beda
            x_bootstrap, y_bootstrap = bootstrapping(X, y)
            # fit
            tree.fit(x_bootstrap, y_bootstrap)
            # simpan ke trees
            self.trees.append(tree)

    def predict(self, X):
        # prediksi di setiap pohon
        tree_prediction = np.array([tree.predict(X) for tree in self.trees]) # [n_tress, n_samples] -> setiap pohon memprediksi seluruh sampel
        tree_prediction = np.swapaxes(tree_prediction, 0, 1) # [n_samples, n_trees] -> 
        # voting
        pred = [most_common_label(tree_pred) for tree_pred in tree_prediction] 
        # kembalikan dalam bentuk array
        return np.array(pred)

