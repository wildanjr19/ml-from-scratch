import numpy as np
from collections import Counter

# entropy
def entropy(y):
    # hitung probabilitas awal dari kelas
    hist = np.bincount(y)
    p_y = hist / len(y)
    return -np.sum([p * np.log2(p) for p in p_y if p > 0]) # menghindari log 0

# kelas node
class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None):
        self.feature = feature          # index fitur untuk split
        self.threshold = threshold      # nilai threshold untuk split
        self.left = left                # node kiri
        self.right = right              # node kanan
        self.value = value              # nilai kelas untuk leaf node

    def is_leaf_node(self):
        return self.value is not None
    

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        """
        Args:
            min_samples_splilt : jumlah minimum sampel yang diperlukan di sebuah node agar node tersebut dapat di split
            max_depth : kedalaman maksimum dari tree -> membatasi pertumbuhan tree
            n_feats : jumlah fitur yang dipilih secara acak untuk dipertimbangkan saat melakukan split
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        """Pertumbuhan pohon / penambahan node"""
        # pastikan fitur
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        # grow
        self.root = self._grow_tree(X, y)

    # helper -> growing tree / penambahan node
    def _grow_tree(self, X, y, depth = 0):
        # insialisasi parameter
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value= leaf_value)
    

        # ambil fitur indeks dari n_features sebanyak n_feats dan tanpa pengembalian (replace=false)
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedy search
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)


    # helper -> mencari kriteria terbaik
    def _best_criteria(self, X, y, feat_idxs):
        # mulai dari nilai -1
        best_gain = -1
        split_idx, split_thresh = None, None

        # looping
        for feat_idx in feat_idxs:
            # ambil semua baris pada tiap feat_idxs
            X_column = X[:, feat_idx]
            # dapatkan threshold dengan mencari nilai uniq di tiap kolom/variabel
            thresholds = np.unique(X_column)



    # helper -> mencari label yang paling sering muncul
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        # melewatkan data di pohon
        pass